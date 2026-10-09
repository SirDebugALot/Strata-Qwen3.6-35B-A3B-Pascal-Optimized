// Exact router payload parity for the event-device-mailbox fence optimization.
// No model needed. Run only when the GPU is free; STRATA_ROUTER_DEVICE_FENCE=1 also exercises production selection.
#include "kernels.cuh"
#include "common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {
constexpr int Layers = 41, Emb = 2048;
struct State {
    sq::HitList hits;
    sq::Mailbox mailbox;
    int32_t nmiss;
    float shared_gate[sq::kMaxT];
    uint32_t counts[Layers * 256];
    unsigned long long ecount[2];
};

template <class T> T* device(size_t n = 1) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&p, n * sizeof(T)));
    return p;
}

void fill_logits(std::vector<float>& logits, int T, int pattern, std::mt19937& rng) {
    for (int t = 0; t < T; ++t) {
        float* row = logits.data() + t * 257;
        for (int i = 0; i < 256; ++i) row[i] = (int(rng() % 20001) - 10000) * 0.015625f;
        switch (pattern) {
        case 0: break; // Random finite values; different expert unions across tokens.
        case 1: std::fill(row, row + 256, 7.0f); break; // Full ties, lower expert ID wins.
        case 2: // Ties spanning lanes/warps with disjoint token-specific subsets.
            std::fill(row, row + 256, -3.0f);
            for (int k = 0; k < 12; ++k) row[(31 * k + 61 * t) % 256] = 9.0f;
            break;
        case 3: // Near ties around 1.0, with signed zeros elsewhere.
            for (int i = 0; i < 256; ++i)
                row[i] = i < 48 ? std::nextafter(1.0f, (i + t) % 3 ? 2.0f : 0.0f) : (i & 1 ? 0.0f : -0.0f);
            break;
        case 4: // Large finite values and negative infinity exercise exponent underflow.
            for (int i = 0; i < 256; ++i)
                row[i] = (i + t) % 19 == 0 ? 1.0e30f : ((i + t) % 7 ? -1.0e30f : -INFINITY);
            break;
        case 5: // NaNs in non-leading lane candidates are ignored by the existing comparison rule.
            // Avoid all-NaN rows: they are outside the router contract and can select duplicate IDs.
            for (int i = 32; i < 256; i += 3) row[i] = std::numeric_limits<float>::quiet_NaN();
            break;
        case 6: // Identical routed experts across tokens exercise repeated pair counts.
            for (int i = 0; i < 256; ++i) row[i] = (float)((i * 17) % 256);
            break;
        }
        row[256] = pattern == 4 ? (t & 1 ? -1000.0f : 1000.0f) : 0.125f * (t - 2);
    }
}

bool oracle(const State& got, const State& initial, const std::vector<float>& logits,
            const std::vector<float>& xn, const std::vector<int32_t>& residency,
            const uint8_t* arena, int T, int layer, uint32_t seq) {
    std::array<std::array<bool, sq::kMaxT>, 256> selected{};
    int hit_pairs = 0, miss_pairs = 0;
    std::vector<uint32_t> counts(std::begin(initial.counts), std::end(initial.counts));
    for (int t = 0; t < T; ++t) {
        std::vector<int> ids;
        for (int e = 0; e < 256; ++e) if (!std::isnan(logits[t * 257 + e])) ids.push_back(e);
        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
            const float x = logits[t * 257 + a], y = logits[t * 257 + b];
            return x > y || (x == y && a < b);
        });
        for (int k = 0; k < sq::kMaxK; ++k) {
            const int e = ids[k];
            selected[e][t] = true;
            ++counts[layer * 256 + e];
            if (residency[layer * 256 + e] >= 0) ++hit_pairs; else ++miss_pairs;
        }
    }
    int nh = 0, nm = 0, pair = 0;
    for (int e = 0; e < 256; ++e) {
        if (std::none_of(selected[e].begin(), selected[e].end(), [](bool v) { return v; })) continue;
        if (residency[layer * 256 + e] >= 0) {
            if (got.hits.ptr[nh] != arena + (size_t)residency[layer * 256 + e] * 16) return false;
            for (int t = 0; t < T; ++t) {
                if (T > 1 && got.hits.pair[nh][t] != (selected[e][t] ? pair : -1)) return false;
                if (selected[e][t]) ++pair;
                else if (got.hits.w[nh][t] != 0.0f) return false;
            }
            ++nh;
        } else {
            if (got.mailbox.ids[nm] != e) return false;
            for (int t = 0; t < T; ++t)
                if (!selected[e][t] && got.mailbox.w[nm][t] != 0.0f) return false;
            ++nm;
        }
    }
    if (got.hits.n != nh || got.hits.n_pair != hit_pairs || got.nmiss != nm || got.mailbox.n_miss != nm ||
        got.mailbox.n_tok != T || got.mailbox.ready_seq != seq ||
        got.ecount[0] != initial.ecount[0] + hit_pairs || got.ecount[1] != initial.ecount[1] + miss_pairs ||
        std::memcmp(got.counts, counts.data(), sizeof(got.counts))) return false;
    if (nm && std::memcmp(got.mailbox.x, xn.data(), (size_t)T * Emb * sizeof(float))) return false;
    return true;
}
} // namespace

int main() {
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaStream_t stream = nullptr;
    cudaEvent_t ready = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    auto* d_state = device<State>();
    auto* d_logits = device<float>(sq::kMaxT * 257);
    auto* d_xn = device<float>(sq::kMaxT * Emb);
    auto* d_residency = device<int32_t>(Layers * 256);
    auto* d_info = device<sq::CacheLayerInfo>(Layers);
    auto* d_params = device<sq::StepParams>();
    auto* arena = device<uint8_t>(256 * 16);
    State *baseline = nullptr, *candidate = nullptr;
    sq::Mailbox *mapped = nullptr, *mapped_device = nullptr;
    CUDA_CHECK(cudaMallocHost((void**)&baseline, sizeof(State)));
    CUDA_CHECK(cudaMallocHost((void**)&candidate, sizeof(State)));
    CUDA_CHECK(cudaHostAlloc((void**)&mapped, sizeof(sq::Mailbox), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&mapped_device, mapped, 0));
    std::vector<sq::CacheLayerInfo> info(Layers, sq::CacheLayerInfo{arena, 16});
    CUDA_CHECK(cudaMemcpyAsync(d_info, info.data(), info.size() * sizeof(info[0]), cudaMemcpyHostToDevice, stream));
    std::mt19937 rng(6100818);
    int cases = 0, comparisons = 0, failures = 0;
    for (int T = 1; T <= sq::kMaxT; ++T) for (int layer : {0, 40})
    for (int pattern = 0; pattern < 7; ++pattern) for (int residency_pattern = 0; residency_pattern < 3; ++residency_pattern) {
        State initial;
        std::memset(&initial, 0xa5, sizeof(initial));
        for (size_t i = 0; i < std::size(initial.counts); ++i) initial.counts[i] = (uint32_t)(i % 7);
        initial.ecount[0] = 17; initial.ecount[1] = 29;
        std::vector<float> logits(T * 257), xn(T * Emb);
        fill_logits(logits, T, pattern, rng);
        for (float& v : xn) v = (int(rng() % 20001) - 10000) * 0.03125f;
        std::vector<int32_t> residency(Layers * 256, -1);
        for (int e = 0; e < 256; ++e)
            if (residency_pattern == 1 || (residency_pattern == 2 && e % 3)) residency[layer * 256 + e] = e;
        sq::StepParams params{};
        params.seq = 0x12345678u + cases; params.n_tok = T;
        CUDA_CHECK(cudaMemcpyAsync(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xn, xn.data(), xn.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_residency, residency.data(), residency.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_params, &params, sizeof(params), cudaMemcpyHostToDevice, stream));
        auto run = [&](State* out, int variant) {
            std::memcpy(out, &initial, sizeof(initial));
            CUDA_CHECK(cudaMemcpyAsync(d_state, &initial, sizeof(initial), cudaMemcpyHostToDevice, stream));
            sq::Mailbox* mailbox = &d_state->mailbox;
            if (variant == 3) { std::memcpy(mapped, &initial.mailbox, sizeof(*mapped)); mailbox = mapped_device; }
            if (variant < 2)
                sq::k_test_router(d_logits, layer, d_residency, d_info, d_xn, &d_state->hits, mailbox, d_params,
                    &d_state->nmiss, d_state->shared_gate, d_state->counts, d_state->ecount, T, stream, variant == 1);
            else
                sq::k_router(d_logits, layer, d_residency, d_info, d_xn, &d_state->hits, mailbox, d_params,
                    &d_state->nmiss, d_state->shared_gate, d_state->counts, d_state->ecount, T, stream, variant == 2);
            CUDA_CHECK(cudaGetLastError());
            // Match the engine's truncated mailbox transfer and host-event visibility protocol.
            if (variant != 3) CUDA_CHECK(cudaMemcpyAsync(&out->mailbox, mailbox,
                offsetof(sq::Mailbox, x) + (size_t)T * sizeof(sq::Mailbox::x[0]), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&out->hits, &d_state->hits, sizeof(out->hits), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&out->nmiss, &d_state->nmiss,
                sizeof(State) - offsetof(State, nmiss), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaEventRecord(ready, stream));
            CUDA_CHECK(cudaEventSynchronize(ready));
            if (variant == 3) std::memcpy(&out->mailbox, mapped, sizeof(*mapped));
        };
        run(baseline, 0);
        if (!oracle(*baseline, initial, logits, xn, residency, arena, T, layer, params.seq)) {
            std::fprintf(stderr, "FAIL host oracle T=%d layer=%d pattern=%d residency=%d\n", T, layer, pattern, residency_pattern);
            ++failures;
        }
        for (int variant = 1; variant <= 3; ++variant) {
            run(candidate, variant);
            if (std::memcmp(baseline, candidate, sizeof(State))) {
                size_t first = 0;
                while (((uint8_t*)baseline)[first] == ((uint8_t*)candidate)[first]) ++first;
                std::fprintf(stderr, "FAIL payload T=%d layer=%d pattern=%d residency=%d variant=%d offset=%zu\n",
                    T, layer, pattern, residency_pattern, variant, first);
                ++failures;
            }
            ++comparisons;
        }
        ++cases;
    }
    CUDA_CHECK(cudaEventDestroy(ready));
    CUDA_CHECK(cudaStreamDestroy(stream));
    for (void* p : {(void*)d_state, (void*)d_logits, (void*)d_xn, (void*)d_residency, (void*)d_info, (void*)d_params, (void*)arena})
        CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFreeHost(baseline)); CUDA_CHECK(cudaFreeHost(candidate)); CUDA_CHECK(cudaFreeHost(mapped));
    std::printf("router mailbox parity: %d cases, %d payload comparisons, %d failures\n", cases, comparisons, failures);
    return failures ? 1 : 0;
}
