// Compare the max-only specialization with the preserved probability path and an independent host oracle.
// No model required. Run only when the GPU is free: test_argmax_no_prob
// The existing all-NaN/all-negative-infinity sentinel is intentionally retained, not treated as a valid token.
#include "kernels.cuh"
#include "common.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {
int reference(const float* row, int n) {
    float best = -INFINITY;
    int id = INT_MAX;
    for (int i = 0; i < n; ++i) {
        // Strict comparison ignores NaNs and -infinity and keeps the first (lowest) tied index.
        if (row[i] > best) { best = row[i]; id = i; }
    }
    return id;
}

void fill(std::vector<float>& x, int n, int T, int pattern, std::mt19937& rng) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int t = 0; t < T; ++t) {
        float* row = x.data() + (size_t)t * n;
        for (int i = 0; i < n; ++i) row[i] = ((int)(rng() % 20001) - 10000) * 0.03125f;
        const int p = (n / 3 + t * 17) % n;
        switch (pattern) {
        case 0: break; // Deterministic random, with naturally occurring finite ties.
        case 1: std::fill(row, row + n, 7.0f); break;
        case 2: // Tied maxima cross warps, blocks, and the final partial block where possible.
            std::fill(row, row + n, -4.0f);
            for (int i : {0, 1, 31, 32, 255, 256, n / 2, n - 1})
                if (i < n) row[i] = 42.0f;
            break;
        case 3:
            for (int i = 0; i < n; i += 3) row[i] = nan;
            row[p] = 1000.0f;
            break;
        case 4: std::fill(row, row + n, nan); break;
        case 5: std::fill(row, row + n, -INFINITY); break;
        case 6:
            std::fill(row, row + n, -INFINITY);
            row[p] = INFINITY; row[n - 1] = INFINITY;
            break;
        case 7:
            std::fill(row, row + n, nan);
            row[p] = -INFINITY; row[n - 1] = INFINITY;
            break;
        case 8:
            for (int i = 0; i < n; ++i) row[i] = i & 1 ? 0.0f : -0.0f;
            break;
        case 9:
            for (int i = 0; i < n; ++i) row[i] = i & 1 ? -1.0e30f : 1.0e30f;
            break;
        }
    }
}
} // namespace

int main() {
    constexpr int max_n = 262145;
    float *logits = nullptr, *probability = nullptr;
    int32_t *with_probability = nullptr, *without_probability = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&logits, (size_t)max_n * sq::kMaxT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&probability, sq::kMaxT * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&with_probability, sq::kMaxT * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void**)&without_probability, sq::kMaxT * sizeof(int32_t)));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::mt19937 rng(6102901);
    int cases = 0, rows = 0, mismatches = 0;
    for (int n : {1, 7, 31, 257, 513, 32767, 248320, max_n}) {
        for (int T = 1; T <= sq::kMaxT; ++T) {
            std::vector<float> x((size_t)n * T);
            for (int pattern = 0; pattern < 10; ++pattern) {
                fill(x, n, T, pattern, rng);
                CUDA_CHECK(cudaMemcpyAsync(logits, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
                // Non-null probability preserves the original kernels' arithmetic and is the device reference.
                sq::k_argmax(logits, n, with_probability, T, stream, probability);
                sq::k_argmax(logits, n, without_probability, T, stream, nullptr);
                int32_t expected[sq::kMaxT], got[sq::kMaxT];
                CUDA_CHECK(cudaMemcpyAsync(expected, with_probability, T * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(got, without_probability, T * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int t = 0; t < T; ++t) {
                    const int oracle = reference(x.data() + (size_t)t * n, n);
                    if (got[t] != expected[t] || got[t] != oracle) {
                        if (mismatches < 20)
                            std::fprintf(stderr, "FAIL n=%d T=%d pattern=%d row=%d old=%d new=%d host=%d\n",
                                         n, T, pattern, t, expected[t], got[t], oracle);
                        ++mismatches;
                    }
                    ++rows;
                }
                ++cases;
            }
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(without_probability));
    CUDA_CHECK(cudaFree(with_probability));
    CUDA_CHECK(cudaFree(probability));
    CUDA_CHECK(cudaFree(logits));
    std::printf("argmax_no_prob: %d cases, %d rows, %d mismatches %s\n", cases, rows, mismatches,
                mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0 ? 0 : 1;
}
