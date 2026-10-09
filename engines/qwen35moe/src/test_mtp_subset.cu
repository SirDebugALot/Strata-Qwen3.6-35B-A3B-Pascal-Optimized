// Bounded standalone GPU regression; no model or server. Parent serializes all GPU runs.
// Checks stratified selection, recent-token inclusion/dedup, bit-exact gathered head rows, and mapped argmax.
#include "mtp_subset.cuh"
#include "common.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {
template <typename T> T* alloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

template <typename T> void upload(T* p, const std::vector<T>& h) {
    CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    // Pageable H2D may return after staging, before DMA completes. The test's consumer stream is nonblocking
    // and does not inherit default-stream ordering, so explicitly finish the upload before using its result.
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

template <typename T> std::vector<T> download(const T* p, size_t n) {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

void run_case(int vocab, int cols, int keep) {
    std::mt19937 rng(261009 + vocab + cols + keep);
    std::vector<int8_t> weights((size_t)vocab * cols), aq(cols);
    std::vector<uint16_t> scales((size_t)vocab * cols / 32);
    std::vector<float> ad(cols / 32), target(vocab);
    for (auto& v : weights) v = (int8_t)((int)(rng() % 255) - 127);
    for (auto& v : scales) {
        const __half h = __float2half((1.0f + (rng() % 19)) / 65536.0f);
        std::memcpy(&v, &h, sizeof(v));
    }
    for (auto& v : aq) v = (int8_t)((int)(rng() % 255) - 127);
    for (auto& v : ad) v = (1.0f + (rng() % 31)) / 4096.0f;
    // Unique, exactly represented ranking keys; a shuffled order covers all vocabulary partitions.
    std::vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    for (int i = 0; i < vocab; ++i) target[i] = (float)(order[i] - vocab / 2) / 8192.0f;

    int8_t* wq = alloc<int8_t>(weights.size());
    uint16_t* wd = alloc<uint16_t>(scales.size());
    sq::ActQ act{alloc<int8_t>(cols), alloc<float>(cols / 32), nullptr};
    float* target_dev = alloc<float>(vocab);
    float* full_dev = alloc<float>(vocab);
    int32_t* token_dev = alloc<int32_t>(1);
    float* prob_dev = alloc<float>(1);
    upload(wq, weights); upload(wd, scales); upload(act.q, aq); upload(act.d, ad);
    const sq::DQ8 head{wq, wd, vocab, cols};
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    sq::k_test_gemv_q8(head, act, full_dev, 1, false, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto full = download(full_dev, vocab);
    size_t checked = 0;
    {
        sq::MtpDraftSubset subset(vocab, keep);
        float* selected_dev = alloc<float>(subset.slots());
        for (int pass = 0; pass < 2; ++pass) {
            if (pass) std::reverse(target.begin(), target.end());
            upload(target_dev, target);
            // Longer than the staging window; include repeated IDs and the highest script-range IDs.
            std::vector<int> history(400);
            for (size_t i = 0; i < history.size(); ++i)
                history[i] = i % 7 ? (vocab - 1 - (int)((i * 13) % vocab) + vocab) % vocab : vocab - 1;
            if (pass) history = {1, vocab - 1, 1}; // reuse must clear the previous pinned staging tail
            const int next = pass ? 0 : vocab - 2;
            subset.prepare(target_dev, history, next, stream);
            subset.project(head, act, selected_dev, token_dev, prob_dev, stream, 0);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto ids = download(subset.ids_device(), subset.slots());
            const auto selected = download(selected_dev, subset.slots());
            const int token = download(token_dev, 1)[0];
            const float prob = download(prob_dev, 1)[0];
            subset.project(head, act, selected_dev, token_dev, prob_dev, stream, 1);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto selected_ro = download(selected_dev, subset.slots());
            const int token_ro = download(token_dev, 1)[0];
            const float prob_ro = download(prob_dev, 1)[0];
            SQ_CHECK(std::memcmp(selected.data(), selected_ro.data(), selected.size() * sizeof(float)) == 0 &&
                     token_ro == token && std::memcmp(&prob, &prob_ro, sizeof(float)) == 0,
                     "read-only gathered head/logit/argmax/probability mismatch");
            const int span = (vocab + sq::MtpDraftSubset::partitions - 1) / sq::MtpDraftSubset::partitions;
            for (int part = 0; part < sq::MtpDraftSubset::partitions; ++part) {
                const int begin = std::min(vocab, part * span), end = std::min(vocab, begin + span);
                std::vector<int> expected(end - begin);
                std::iota(expected.begin(), expected.end(), begin);
                std::sort(expected.begin(), expected.end(), [&](int a, int b) { return target[a] > target[b]; });
                for (int i = 0; i < keep; ++i) {
                    const int wanted = i < (int)expected.size() ? expected[i] : -1;
                    const int actual = ids[part * keep + i];
                    SQ_CHECK(actual == wanted,
                             "partition rank mismatch vocab=%d keep=%d pass=%d partition=%d rank=%d actual=%d (%.9g) expected=%d (%.9g)",
                             vocab, keep, pass, part, i, actual, actual >= 0 && actual < vocab ? target[actual] : -INFINITY,
                             wanted, wanted >= 0 ? target[wanted] : -INFINITY);
                }
            }
            const int base = sq::MtpDraftSubset::partitions * keep;
            const int recent = (int)std::min(history.size(), (size_t)255);
            for (int i = 0; i < recent; ++i)
                SQ_CHECK(ids[base + i] == history[history.size() - recent + i], "recent history window mismatch");
            SQ_CHECK(ids[base + recent] == next, "next token missing from recent window");
            for (int i = recent + 1; i < 256; ++i) SQ_CHECK(ids[base + i] == -1, "stale recent staging tail");
            std::vector<bool> seen(vocab, false);
            int best = -1;
            for (int i = 0; i < subset.slots(); ++i) {
                const int id = ids[i];
                const bool valid = id >= 0 && id < vocab && !seen[id];
                if (valid) {
                    seen[id] = true;
                    SQ_CHECK(std::isfinite(selected[i]) && std::memcmp(&selected[i], &full[id], sizeof(float)) == 0,
                             "gathered head row differs: vocab=%d cols=%d keep=%d pass=%d slot=%d token=%d subset=%.9g full=%.9g",
                             vocab, cols, keep, pass, i, id, selected[i], full[id]);
                    if (best < 0 || selected[i] > selected[best]) best = i;
                    ++checked;
                } else {
                    SQ_CHECK(std::isinf(selected[i]) && selected[i] < 0, "invalid/duplicate row contributes to softmax");
                }
            }
            SQ_CHECK(best >= 0 && token == ids[best], "subset argmax ID mapping mismatch");
            double denominator = 0;
            for (float value : selected) denominator += std::exp((double)value - selected[best]);
            const double expected_prob = 1.0 / denominator;
            SQ_CHECK(std::isfinite(prob) && std::abs(prob - expected_prob) <= 2e-5 * expected_prob,
                     "subset probability mismatch: %.9g vs %.9g", prob, expected_prob);
            if (span <= keep) {
                SQ_CHECK(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }), "small vocabulary lost entries");
                double full_denominator = 0;
                const float maximum = *std::max_element(full.begin(), full.end());
                for (float value : full) full_denominator += std::exp((double)value - maximum);
                SQ_CHECK(std::abs(expected_prob - 1.0 / full_denominator) < 1e-12, "full-coverage probability differs");
            }
        }
        CUDA_CHECK(cudaFree(selected_dev));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(wq)); CUDA_CHECK(cudaFree(wd));
    CUDA_CHECK(cudaFree(act.q)); CUDA_CHECK(cudaFree(act.d));
    CUDA_CHECK(cudaFree(target_dev)); CUDA_CHECK(cudaFree(full_dev));
    CUDA_CHECK(cudaFree(token_dev)); CUDA_CHECK(cudaFree(prob_dev));
    std::printf("vocab=%d cols=%d per_partition=%d passes=2 exact_rows=%zu PASS\n", vocab, cols, keep, checked);
}
} // namespace

int main() {
    run_case(257, 96, 32); // padded final partitions, short vector, and all-vocabulary coverage
    run_case(4096, 2048, 64); // the real Qwen hidden width, with invalid and duplicate slots
    for (int keep : {32, 64, 128}) run_case(248320, 256, keep); // actual vocabulary, including high script IDs
    run_case(262144, 32, 64); // the maximum supported partition size and minimal quantized row
    std::puts("MTP SUBSET SELECTION / EXACT HEAD ROWS / ARGMAX PASS");
    return 0;
}
