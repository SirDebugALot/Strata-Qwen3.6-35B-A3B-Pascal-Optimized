#include "cpu_lm_head.hpp"
#include "cpu_moe.hpp"
#include "quant.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    if (!sq::cpu_has_avx2()) {
        std::puts("SKIP: CPU LM head needs AVX2/FMA/F16C");
        return 0;
    }
    std::mt19937 rng(61026);
    constexpr int workers = 6;
    sq::CpuMoe pool;
    pool.start(workers, 0, {}, nullptr, nullptr, false);  // CPU-only jobs: no model, mailbox or CUDA allocation.
    std::array<std::thread::id, workers> worker_ids;
    pool.parallel_for_workers([&](int worker) { worker_ids[worker] = std::this_thread::get_id(); });
    std::array<std::atomic<int>, workers> calls{};
    std::atomic<bool> stable_workers{true};
    for (int job = 0; job < 128; ++job)
        pool.parallel_for_workers([&](int worker) {
            if (worker_ids[worker] != std::this_thread::get_id()) stable_workers = false;
            ++calls[worker];
        });
    for (int worker = 0; worker < workers; ++worker) {
        if (!stable_workers || calls[worker] != 128) {
            std::fputs("CPU LM head pool changed workers or missed/repeated a job\n", stderr);
            return 1;
        }
    }
    double max_error = 0.0;
    for (int cols : {32, 64, 2048}) {
        const int rows = 37, nb = cols / 32;
        std::vector<int8_t> weights((size_t)rows * cols), act(cols);
        std::vector<uint16_t> scales((size_t)rows * nb);
        std::vector<float> act_scales(nb), out(rows, -9999.0f);
        for (auto& x : weights) x = (int8_t)((int)(rng() % 256) - 128);
        for (auto& x : act) x = (int8_t)((int)(rng() % 256) - 128);
        // The extremes catch accidental use of saturating unsigned/signed int8 dot products.
        for (int i = 0; i < 32; ++i) { weights[i] = -128; act[i] = -128; }
        for (auto& x : scales) x = sq::f32_to_fp16((float)(1 + rng() % 64) / 256.0f);
        for (auto& x : act_scales) x = (float)(1 + rng() % 64) / 512.0f;
        sq::cpu_lm_head_q8_rows(weights.data(), scales.data(), act.data(), act_scales.data(),
                                out.data(), cols, 0, 13);
        if (out[13] != -9999.0f || out.back() != -9999.0f) {
            std::fputs("CPU LM head wrote outside its assigned rows\n", stderr);
            return 1;
        }
        sq::cpu_lm_head_q8_rows(weights.data(), scales.data(), act.data(), act_scales.data(),
                                out.data(), cols, 13, rows);
        // The engine uses this same persistent pool partition for every row of a verification window.
        // The last-row-only case must leave the other output rows untouched.
        for (int nt : {1, 2, 3, 4}) {
            std::vector<float> pooled((size_t)nt * rows, -9999.0f);
            pool.parallel_for_workers([&](int worker) {
                const int first = rows * worker / workers, last = rows * (worker + 1) / workers;
                for (int t = 0; t < nt; ++t)
                    sq::cpu_lm_head_q8_rows(weights.data(), scales.data(), act.data(), act_scales.data(),
                                            pooled.data() + (size_t)t * rows, cols, first, last);
            });
            for (int t = 0; t < nt; ++t)
                if (!std::equal(out.begin(), out.end(), pooled.begin() + (size_t)t * rows)) {
                    std::fputs("CPU LM head pool projection differs from direct rows\n", stderr);
                    return 1;
                }
            std::fill(pooled.begin(), pooled.end(), -9999.0f);
            pool.parallel_for_workers([&](int worker) {
                sq::cpu_lm_head_q8_rows(weights.data(), scales.data(), act.data(), act_scales.data(),
                                        pooled.data() + (size_t)(nt - 1) * rows, cols,
                                        rows * worker / workers, rows * (worker + 1) / workers);
            });
            for (int i = 0; i < (nt - 1) * rows; ++i)
                if (pooled[i] != -9999.0f) return 1;
            if (!std::equal(out.begin(), out.end(), pooled.begin() + (size_t)(nt - 1) * rows)) return 1;
        }
        // Distinct tokens expose accidental activation/scale stride reuse. Include a zero token and
        // signed int8 extremes, then demand bitwise T=1 parity, including rows outside a worker range.
        std::vector<int8_t> multi_act((size_t)4 * cols);
        std::vector<float> multi_scale((size_t)4 * nb);
        for (auto& x : multi_act) x = (int8_t)((int)(rng() % 256) - 128);
        for (auto& x : multi_scale) x = (float)(1 + rng() % 97) / 768.0f;
        for (int i = 0; i < cols; ++i) {
            multi_act[cols + i] = (i & 1) ? 127 : -128;
            multi_act[2 * cols + i] = 0;
        }
        std::fill(multi_scale.begin() + 2 * nb, multi_scale.begin() + 3 * nb, 0.0f);
        for (int nt = 1; nt <= 4; ++nt) {
            std::vector<float> expected((size_t)nt * rows, -9999.0f), batched(expected);
            for (int t = 0; t < nt; ++t)
                sq::cpu_lm_head_q8_rows(weights.data(), scales.data(), multi_act.data() + (size_t)t * cols,
                                        multi_scale.data() + (size_t)t * nb, expected.data() + (size_t)t * rows,
                                        cols, 1, rows - 1);
            sq::cpu_lm_head_q8_rows_multi(weights.data(), scales.data(), multi_act.data(), multi_scale.data(),
                                          batched.data(), cols, rows, 1, rows - 1, nt);
            if (std::memcmp(expected.data(), batched.data(), expected.size() * sizeof(float)) != 0) {
                std::fprintf(stderr, "CPU LM head multi-token bitwise mismatch: cols=%d nt=%d (partial range)\n", cols, nt);
                return 1;
            }
            for (int t = 0; t < nt; ++t)
                sq::cpu_lm_head_q8_rows(weights.data(), scales.data(), multi_act.data() + (size_t)t * cols,
                                        multi_scale.data() + (size_t)t * nb, expected.data() + (size_t)t * rows,
                                        cols, 0, rows);
            std::fill(batched.begin(), batched.end(), -9999.0f);
            pool.parallel_for_workers([&](int worker) {
                sq::cpu_lm_head_q8_rows_multi(weights.data(), scales.data(), multi_act.data(), multi_scale.data(),
                                              batched.data(), cols, rows,
                                              rows * worker / workers, rows * (worker + 1) / workers, nt);
            });
            if (std::memcmp(expected.data(), batched.data(), expected.size() * sizeof(float)) != 0) {
                std::fprintf(stderr, "CPU LM head multi-token bitwise mismatch: cols=%d nt=%d (pooled)\n", cols, nt);
                return 1;
            }
            // Empty ranges must perform no writes even for all four tokens.
            sq::cpu_lm_head_q8_rows_multi(weights.data(), scales.data(), multi_act.data(), multi_scale.data(),
                                          batched.data(), cols, rows, 7, 7, nt);
            if (std::memcmp(expected.data(), batched.data(), expected.size() * sizeof(float)) != 0) return 1;
        }
        for (int row = 0; row < rows; ++row) {
            double ref = 0.0, abs_sum = 0.0;
            for (int col = 0; col < cols; ++col) {
                const double w = (double)weights[(size_t)row * cols + col] *
                                 sq::fp16_to_f32(scales[(size_t)row * nb + col / 32]);
                const double a = (double)act[col] * act_scales[col / 32];
                ref += w * a;
                abs_sum += std::abs(w * a);
            }
            const double error = std::abs((double)out[row] - ref);
            max_error = std::max(max_error, error);
            if (!std::isfinite(out[row]) || error > 2e-6 * (1.0 + abs_sum)) {
                std::fprintf(stderr, "CPU LM head mismatch: cols=%d row=%d expected=%.9g actual=%.9g\n",
                             cols, row, ref, out[row]);
                return 1;
            }
        }
    }
    std::printf("CPU LM head tests passed (T=1..4 bitwise batch parity, persistent pool, row bounds; max absolute error %.9g)\n", max_error);
    return 0;
}
