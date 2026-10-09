// Full resident-MoE parity: explicit old/new launchers, shared and sparse expert/token routes.
// Link with sq_core. No model file, cache mutation or server is required. Parent serializes GPU runs.
#include "kernels.cuh"
#include "common.hpp"
#include "quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
template <typename T> T* alloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

sq::ActQ alloc_act(int n) { return {alloc<int8_t>(n), alloc<float>(n / 32), alloc<float>(n / 32)}; }
void free_act(sq::ActQ a) { CUDA_CHECK(cudaFree(a.q)); CUDA_CHECK(cudaFree(a.d)); CUDA_CHECK(cudaFree(a.s)); }

size_t compare_float(const char* stage, uint32_t type, int nt, int nh, int pattern,
                     const float* baseline, const float* reused, size_t n) {
    std::vector<float> a(n), b(n);
    CUDA_CHECK(cudaMemcpy(a.data(), baseline, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), reused, n * sizeof(float), cudaMemcpyDeviceToHost));
    size_t different = 0;
    double max_delta = 0, sq_delta = 0, sq_ref = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)b[i] - a[i];
        max_delta = std::max(max_delta, std::abs(d));
        sq_delta += d * d;
        sq_ref += (double)a[i] * a[i];
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::memcmp(&a[i], &b[i], sizeof(float))) {
            if (different < 2)
                std::fprintf(stderr, "%s %s NT=%d hits=%d pattern=%d index=%zu old=%.9g reuse=%.9g\n",
                             stage, sq::type_name(type), nt, nh, pattern, i, a[i], b[i]);
            ++different;
        }
    }
    if (different)
        std::fprintf(stderr, "%s %s NT=%d hits=%d pattern=%d: %zu/%zu differing; maxabs=%.9g relativeL2=%.9g\n",
                     stage, sq::type_name(type), nt, nh, pattern, different, n, max_delta,
                     std::sqrt(sq_delta / std::max(sq_ref, 1e-30)));
    return different;
}
}  // namespace

int main() {
#ifdef SQ_TEST_NATIVE_LAYOUT
    constexpr bool layout_test = true;
#else
    constexpr bool layout_test = false;
#endif
    constexpr int E = 2048, FF = 512, MAX_PAIR = sq::kMaxT * sq::kMaxK;
    std::mt19937 rng(0x6f13);
    sq::ActQ x = alloc_act(sq::kMaxT * E);
    sq::ActQ hq[2] = {alloc_act(MAX_PAIR * FF), alloc_act(MAX_PAIR * FF)};
    float* gu[2] = {alloc<float>(MAX_PAIR * 2 * FF), alloc<float>(MAX_PAIR * 2 * FF)};
    float* down[2] = {alloc<float>(sq::kMaxT * E), alloc<float>(sq::kMaxT * E)};
    sq::HitList* hit_dev = alloc<sq::HitList>(1);
    std::vector<int8_t> aq(sq::kMaxT * E);
    std::vector<float> ad(sq::kMaxT * E / 32), as(ad.size());
    for (auto& q : aq) q = (int8_t)((int)(rng() % 255) - 127);
    for (size_t b = 0; b < ad.size(); ++b) {
        ad[b] = (1.0f + (rng() % 31)) / 4096.0f;
        int sum = 0;
        for (int j = 0; j < 32; ++j) sum += aq[b * 32 + j];
        as[b] = sum * ad[b];
    }
    CUDA_CHECK(cudaMemcpy(x.q, aq.data(), aq.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x.d, ad.data(), ad.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x.s, as.data(), as.size() * sizeof(float), cudaMemcpyHostToDevice));
    size_t failures = 0, cases = 0;
    for (uint32_t type : {sq::T_IQ2_S, sq::T_IQ3_S, sq::T_IQ4_XS, sq::T_Q2_K, sq::T_Q3_K}) {
        const size_t block_bytes = (size_t)sq::row_bytes(type, 256);
        const size_t gate_bytes = sq::row_bytes(type, E) * FF;
        const size_t blob_bytes = 2 * gate_bytes + sq::row_bytes(type, FF) * E;
        std::vector<uint8_t> blob(blob_bytes * sq::kMaxK);
        for (auto& b : blob) b = (uint8_t)rng();
        const int d_offset = type == sq::T_Q2_K ? 80 : type == sq::T_Q3_K ? 108 : 0;
        for (size_t b = 0; b < blob.size(); b += block_bytes) {
            const uint16_t d = sq::f32_to_fp16((1.0f + rng() % 7) / 4096.0f);
            std::memcpy(blob.data() + b + d_offset, &d, sizeof(d));
            if (type == sq::T_Q2_K) {
                const uint16_t dm = sq::f32_to_fp16((1.0f + rng() % 5) / 4096.0f);
                std::memcpy(blob.data() + b + 82, &dm, sizeof(dm));
            }
        }
        uint8_t* weights = alloc<uint8_t>(blob.size());
        CUDA_CHECK(cudaMemcpy(weights, blob.data(), blob.size(), cudaMemcpyHostToDevice));
        for (int reuse_mode = layout_test ? 0 : 1; reuse_mode <= 1; ++reuse_mode)
        for (int nt : {1, 2, 3, 4}) for (int nh : {1, 3, 8}) for (int pattern = 0; pattern < 3; ++pattern) {
            if (!layout_test && nt == 1) continue;
            const bool old_reuse = layout_test && reuse_mode, new_reuse = reuse_mode != 0;
            sq::HitList hits{};
            hits.n = nh;
            std::memset(hits.pair, -1, sizeof(hits.pair));
            for (int j = 0; j < nh; ++j) {
                hits.ptr[j] = weights + j * blob_bytes;
                unsigned mask = pattern == 0 ? (1u << nt) - 1 :
                                pattern == 1 ? 1u << (j % nt) : (j * 5u + 3u) & ((1u << nt) - 1);
                if (!mask) mask = 1u << (j % nt);
                for (int t = 0; t < nt; ++t) if (mask & (1u << t)) {
                    hits.pair[j][t] = (int8_t)hits.n_pair++;
                    hits.w[j][t] = (float)(j + 1) / 36.0f;
                }
            }
            CUDA_CHECK(cudaMemcpy(hit_dev, &hits, sizeof(hits), cudaMemcpyHostToDevice));
            sq::k_test_moe_gu(type, x, hit_dev, gu[0], nt, old_reuse, nullptr);
            sq::k_test_moe_gu(type, x, hit_dev, gu[1], nt, new_reuse, nullptr, layout_test);
            CUDA_CHECK(cudaGetLastError());
            failures += compare_float("gate/up", type, nt, nh, pattern, gu[0], gu[1], (size_t)hits.n_pair * 2 * FF);
            sq::k_moe_act(gu[0], hit_dev, hq[0], nt, nullptr);
            sq::k_moe_act(gu[1], hit_dev, hq[1], nt, nullptr);
            CUDA_CHECK(cudaGetLastError());
            failures += compare_float("activation scale", type, nt, nh, pattern, hq[0].d, hq[1].d,
                                      (size_t)hits.n_pair * FF / 32);
            std::vector<int8_t> act0((size_t)hits.n_pair * FF), act1(act0.size());
            CUDA_CHECK(cudaMemcpy(act0.data(), hq[0].q, act0.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(act1.data(), hq[1].q, act1.size(), cudaMemcpyDeviceToHost));
            if (act0 != act1) { std::fprintf(stderr, "activation bytes differ for %s NT=%d hits=%d pattern=%d\n",
                                            sq::type_name(type), nt, nh, pattern); ++failures; }
            // First hold the activation identical to isolate the down kernel, then test the whole chain.
            sq::k_test_moe_down(type, 2 * gate_bytes, hq[0], hit_dev, down[0], nt, old_reuse, nullptr);
            sq::k_test_moe_down(type, 2 * gate_bytes, hq[0], hit_dev, down[1], nt, new_reuse, nullptr, layout_test);
            CUDA_CHECK(cudaGetLastError());
            failures += compare_float("down isolated", type, nt, nh, pattern, down[0], down[1], (size_t)nt * E);
            sq::k_test_moe_down(type, 2 * gate_bytes, hq[1], hit_dev, down[1], nt, new_reuse, nullptr, layout_test);
            CUDA_CHECK(cudaGetLastError());
            failures += compare_float("full MoE", type, nt, nh, pattern, down[0], down[1], (size_t)nt * E);
            if (layout_test && pattern != 1 && nh != 1) {
                // Bounded CUDA-event microbenchmark, not an end-to-end speed claim. Alternate trial order to
                // avoid systematically favoring either warm cache/clock state, and report medians of three.
                cudaEvent_t start, end;
                CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&end));
                float elapsed[2][3] = {};
                constexpr int repetitions = 32;
                for (int round = 0; round < 3; ++round) for (int order = 0; order < 2; ++order) {
                    const int variant = order ^ (round & 1);
                    const bool transpose = variant != 0, reuse = variant ? new_reuse : old_reuse;
                    for (int warm = 0; warm < 2; ++warm) {
                        sq::k_test_moe_gu(type, x, hit_dev, gu[variant], nt, reuse, nullptr, transpose);
                        sq::k_moe_act(gu[variant], hit_dev, hq[variant], nt, nullptr);
                        sq::k_test_moe_down(type, 2 * gate_bytes, hq[variant], hit_dev, down[variant], nt, reuse, nullptr, transpose);
                    }
                    CUDA_CHECK(cudaEventRecord(start));
                    for (int rep = 0; rep < repetitions; ++rep) {
                        sq::k_test_moe_gu(type, x, hit_dev, gu[variant], nt, reuse, nullptr, transpose);
                        sq::k_moe_act(gu[variant], hit_dev, hq[variant], nt, nullptr);
                        sq::k_test_moe_down(type, 2 * gate_bytes, hq[variant], hit_dev, down[variant], nt, reuse, nullptr, transpose);
                    }
                    CUDA_CHECK(cudaEventRecord(end));
                    CUDA_CHECK(cudaEventSynchronize(end));
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaEventElapsedTime(&elapsed[variant][round], start, end));
                }
                std::sort(elapsed[0], elapsed[0] + 3);
                std::sort(elapsed[1], elapsed[1] + 3);
                std::printf("LAYOUT %s NT=%d hits=%d pattern=%d reuse=%d baseline_us=%.3f transpose_us=%.3f speedup=%.4f\n",
                            sq::type_name(type), nt, nh, pattern, reuse_mode,
                            elapsed[0][1] * 1000 / repetitions, elapsed[1][1] * 1000 / repetitions,
                            elapsed[0][1] / elapsed[1][1]);
                CUDA_CHECK(cudaEventDestroy(start));
                CUDA_CHECK(cudaEventDestroy(end));
            }
            ++cases;
        }
        CUDA_CHECK(cudaFree(weights));
    }
    CUDA_CHECK(cudaFree(hit_dev));
    free_act(x);
    for (int i = 0; i < 2; ++i) { free_act(hq[i]); CUDA_CHECK(cudaFree(gu[i])); CUDA_CHECK(cudaFree(down[i])); }
    std::printf("GPU resident expert %s: %zu cases, %zu differing values; bitwise parity %s\n",
                layout_test ? "transposed activation layout" : "reuse", cases, failures, failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
