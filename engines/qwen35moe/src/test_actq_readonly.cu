// Exact old/read-only dense staging comparison; no model or server.
#include "kernels.cuh"
#include "common.hpp"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <vector>

template<class V> V* alloc(size_t n) {
    V* p = nullptr; CUDA_CHECK(cudaMalloc(&p, n * sizeof(V))); return p;
}
template<class V> void upload(V* p, const std::vector<V>& v, cudaStream_t s) {
    CUDA_CHECK(cudaMemcpyAsync(p, v.data(), v.size() * sizeof(V), cudaMemcpyHostToDevice, s));
}
template<class V> std::vector<V> read(V* p, size_t n, cudaStream_t s) {
    std::vector<V> v(n);
    CUDA_CHECK(cudaMemcpyAsync(v.data(), p, n * sizeof(V), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s)); return v;
}

void check(int rows, int cols) {
    constexpr int Guard = 16;
    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    std::vector<int8_t> w((size_t)rows * cols), q((size_t)5 * cols);
    std::vector<uint16_t> wd((size_t)rows * cols / 32);
    std::vector<float> d((size_t)5 * cols / 32);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (int8_t)((int)((i * 31 + i / cols) & 255) - 128);
    for (size_t i = 0; i < wd.size(); ++i) {
        __half h = __float2half((1 + (i % 19)) / 65536.0f);
        std::memcpy(&wd[i], &h, sizeof(h));
    }
    auto* wq = alloc<int8_t>(w.size()); auto* ws = alloc<uint16_t>(wd.size());
    sq::ActQ a{alloc<int8_t>(q.size()), alloc<float>(d.size()), nullptr};
    auto* old = alloc<float>((size_t)4 * rows + 2 * Guard);
    auto* ro = alloc<float>((size_t)4 * rows + 2 * Guard);
    upload(wq, w, s); upload(ws, wd, s);
    sq::DQ8 head{wq, ws, rows, cols};
    size_t compared = 0;
    for (int pass = 0; pass < 3; ++pass) {
        for (size_t i = 0; i < q.size(); ++i) q[i] = pass == 0 ? 0 : (int8_t)((int)((i * 17 + pass * 53) & 255) - 128);
        for (size_t i = 0; i < d.size(); ++i) d[i] = (1 + ((i * 7 + pass) % 31)) / 4096.0f;
        upload(a.q, q, s); upload(a.d, d, s);
        // Offset one row: exercises the same aligned final-row view used by MTP.
        // GEMV never reads s; avoid ActQ::row doing arithmetic on that null member.
        sq::ActQ view{a.q + cols, a.d + cols / 32, nullptr};
        for (int T = 1; T <= 4; ++T) {
            const size_t count = (size_t)T * rows + 2 * Guard;
            CUDA_CHECK(cudaMemsetAsync(old, 0x5a, count * sizeof(float), s));
            CUDA_CHECK(cudaMemsetAsync(ro, 0x5a, count * sizeof(float), s));
            sq::k_test_gemv_q8(head, view, old + Guard, T, false, s);
            sq::k_test_gemv_q8(head, view, ro + Guard, T, true, s);
            CUDA_CHECK(cudaGetLastError());
            const auto expected = read(old, count, s), actual = read(ro, count, s);
            SQ_CHECK(std::memcmp(expected.data(), actual.data(), count * sizeof(float)) == 0,
                     "read-only mismatch rows=%d cols=%d T=%d pass=%d", rows, cols, T, pass);
            uint32_t marker = 0x5a5a5a5a;
            for (int i = 0; i < Guard; ++i)
                SQ_CHECK(!std::memcmp(&actual[i], &marker, sizeof(marker)) &&
                         !std::memcmp(&actual[count - Guard + i], &marker, sizeof(marker)), "output guard changed");
            compared += (size_t)T * rows;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(wq)); CUDA_CHECK(cudaFree(ws)); CUDA_CHECK(cudaFree(a.q)); CUDA_CHECK(cudaFree(a.d));
    CUDA_CHECK(cudaFree(old)); CUDA_CHECK(cudaFree(ro)); CUDA_CHECK(cudaStreamDestroy(s));
    std::printf("rows=%d cols=%d exact_values=%zu PASS\n", rows, cols, compared);
}
int main() {
    check(1, 32); check(17, 96); check(512, 2048); check(12288, 2048);
    check(2048, 4096); check(248320, 2048);
    std::puts("ACTQ READ-ONLY DENSE STAGING / NT1..4 / REWRITES / OFFSET / GUARDS PASS");
}
