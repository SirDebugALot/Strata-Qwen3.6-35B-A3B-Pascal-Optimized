// Lossless, row-local Q6_K byte rearrangement for coalesced 32-bit GPU loads.
// This header is CPU-only so the loader and unit fixtures share the exact size/storage contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sq {

inline size_t dense_q6_plane_row_bytes(int cols) {
    return ((size_t)(cols / 256) * 210 + 31) & ~(size_t)31;
}

inline void dense_q6_pack_row_planes(const uint8_t* source, uint8_t* destination, int cols) {
    const size_t blocks = (size_t)cols / 256;
    // Padding is deterministic; all 210 source bytes per block are copied without converting their values.
    std::memset(destination, 0, dense_q6_plane_row_bytes(cols));
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* input = source + block * 210;
        for (size_t word = 0; word < 8; ++word) {
            for (size_t segment = 0; segment < 4; ++segment)
                std::memcpy(destination + 4 * (word * blocks * 4 + block * 4 + segment),
                            input + segment * 32 + word * 4, 4);
            for (size_t half = 0; half < 2; ++half)
                std::memcpy(destination + blocks * 128 + 4 * (word * blocks * 2 + block * 2 + half),
                            input + 128 + half * 32 + word * 4, 4);
        }
        std::memcpy(destination + blocks * 192 + block * 16, input + 192, 16);
        std::memcpy(destination + blocks * 208 + block * 2, input + 208, 2);
    }
}

} // namespace sq
