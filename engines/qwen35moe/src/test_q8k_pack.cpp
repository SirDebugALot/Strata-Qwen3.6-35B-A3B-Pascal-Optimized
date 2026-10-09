// CPU-only exact Q8K packing parity. Optional --bench measures warmed small activations on this CPU.
#include "common.hpp"
#include "cpu_kernels.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    SQ_CHECK(sq::cpu_force_isa("avx2"), "Q8K packing test requires AVX2/FMA/F16C");
    std::mt19937 rng(61029);
    std::vector<float> x(8192);
    std::vector<sq::Q8K> scalar(x.size() / 256), packed(scalar.size());
    const unsigned saved = _mm_getcsr();
    size_t blocks = 0;
    for (unsigned rounding : {_MM_ROUND_NEAREST, _MM_ROUND_DOWN, _MM_ROUND_UP, _MM_ROUND_TOWARD_ZERO}) {
        _mm_setcsr((saved & ~_MM_ROUND_MASK) | rounding);
        for (int pattern = 0; pattern < 68; ++pattern) {
            for (size_t i = 0; i < x.size(); ++i) {
                switch (pattern) {
                    case 0: x[i] = 0; break;
                    case 1: x[i] = (i & 1) ? -0.0f : 0.0f; break;
                    case 2: x[i] = (i & 1) ? -1.0f : 1.0f; break;
                    case 3: x[i] = (float)((int)(i % 255) - 127) + 0.5f; break;
                    case 4: x[i] = (i % 256 == 0) ? 127.0f : (float)((int)(i % 253) - 126) + 0.5f; break;
                    case 5: x[i] = std::nextafter((float)((int)(i % 255) - 127) + 0.5f, 0.0f); break;
                    case 6: x[i] = (i & 1) ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max(); break;
                    case 7: x[i] = (i & 1) ? std::numeric_limits<float>::min() : -std::numeric_limits<float>::min(); break;
                    default: x[i] = std::ldexp((float)((int)(rng() % 2000001) - 1000000) / 1000000.0f,
                                              (pattern % 31) - 15); break;
                }
            }
            // Q8K has trailing alignment padding. Initialize it identically; compare the complete record,
            // including scales, natural/permuted bytes, sub-block sums and untouched padding.
            std::memset(scalar.data(), 0x5a, scalar.size() * sizeof(sq::Q8K));
            std::memset(packed.data(), 0x5a, packed.size() * sizeof(sq::Q8K));
            sq::avx2::quantize_q8k_test_pack(x.data(), scalar.data(), (int)x.size(), false);
            sq::avx2::quantize_q8k_test_pack(x.data(), packed.data(), (int)x.size(), true);
            SQ_CHECK(std::memcmp(scalar.data(), packed.data(), scalar.size() * sizeof(sq::Q8K)) == 0,
                     "Q8K packing mismatch: rounding=%u pattern=%d", rounding, pattern);
            blocks += scalar.size();
        }
    }
    _mm_setcsr(saved);
    sq::log("Q8K vector packing: %zu blocks, all MXCSR rounding modes, complete-record bitwise PASS", blocks);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        for (int n : {512, 2048, 8192}) {
            double ms[2][3] = {};
            constexpr int repetitions = 4000;
            for (int round = 0; round < 3; ++round) for (int order = 0; order < 2; ++order) {
                const int variant = order ^ (round & 1);
                for (int warm = 0; warm < 16; ++warm)
                    sq::avx2::quantize_q8k_test_pack(x.data(), packed.data(), n, variant != 0);
                const double start = sq::now_ms();
                for (int rep = 0; rep < repetitions; ++rep)
                    sq::avx2::quantize_q8k_test_pack(x.data(), packed.data(), n, variant != 0);
                ms[variant][round] = sq::now_ms() - start;
            }
            std::sort(ms[0], ms[0] + 3);
            std::sort(ms[1], ms[1] + 3);
            sq::log("Q8K packing n=%d scalar_ns=%.1f packed_ns=%.1f speedup=%.3f", n,
                    ms[0][1] * 1e6 / repetitions, ms[1][1] * 1e6 / repetitions, ms[0][1] / ms[1][1]);
        }
    }
    return 0;
}
