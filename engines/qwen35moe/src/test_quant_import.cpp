// Compare sampled real GGUF blocks with independent gguf-py reference output.
// Fixture: repeated {uint32 type, uint32 elements, packed weights, float32 reference[elements]}.
#include "quant.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) { std::fprintf(stderr, "usage: test_quant_import fixture.bin [--fast]\n"); return 2; }
    const bool fast = argc == 3 && std::strcmp(argv[2], "--fast") == 0;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    int cases = 0;
    double error = 0, energy = 0, max_delta = 0;
    uint32_t header[2];
    while (std::fread(header, sizeof(header), 1, f) == 1) {
        const uint32_t type = header[0], n = header[1];
        if (!sq::can_dequant(type) || !n || n % 256 || n > 65536) return 3;
        std::vector<uint8_t> packed((size_t)sq::row_bytes(type, n));
        std::vector<float> expected(n), actual(n), converted(n);
        std::vector<sq::BlockQ6_K> q6(n / 256);
        if (std::fread(packed.data(), packed.size(), 1, f) != 1 ||
            std::fread(expected.data(), sizeof(float) * n, 1, f) != 1) return 3;
        sq::dequant_row(type, packed.data(), actual.data(), n);
        double e = 0, v = 0, max_diff = 0;
        for (uint32_t j = 0; j < n; ++j) {
            if (!std::isfinite(actual[j]) || !std::isfinite(expected[j])) return 4;
            max_diff = std::max(max_diff, std::fabs((double)actual[j] - expected[j]));
            if (std::fabs(actual[j] - expected[j]) > 1e-6f + 1e-5f * std::fabs(expected[j])) {
                std::fprintf(stderr, "%s case %d element %u: %g != %g\n", sq::type_name(type), cases, j, actual[j], expected[j]);
                return 4;
            }
        }
        sq::quantize_row_q6_K(actual.data(), q6.data(), n, fast);
        sq::dequant_q6_K(q6.data(), converted.data(), n);
        for (uint32_t j = 0; j < n; ++j) {
            const double d = (double)converted[j] - actual[j];
            e += d * d;
            v += (double)actual[j] * actual[j];
        }
        error += e;
        energy += v;
        max_delta = std::max(max_delta, max_diff);
        ++cases;
    }
    std::fclose(f);
    const double relative_rms = std::sqrt(error / std::max(energy, 1e-30));
    std::printf("%d real-block cases matched gguf-py, max absolute decoder difference %.9g; Q6_K conversion relative RMS %.6f\n",
                cases, max_delta, relative_rms);
    return cases && relative_rms < 0.04 ? 0 : 5;
}
