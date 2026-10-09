// Exact original/prepared GDN recurrence comparison. No model/cache/server required.
// Parent serializes this GPU test with all other GPU work.
#include "kernels.cuh"
#include "common.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int Heads = 32, Dim = 128, Conv = 8192, Out = 4096, Guard = 16;
constexpr size_t State = (size_t)Heads * Dim * Dim;
constexpr float Sentinel = 123456.25f;
uint64_t checks = 0, cases = 0;

struct Buffer {
    float* p = nullptr;
    size_t n;
    explicit Buffer(size_t size) : n(size) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(float))); }
    ~Buffer() { if (p) cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<float>& v) {
        SQ_CHECK(v.size() == n, "fixture upload extent");
        CUDA_CHECK(cudaMemcpy(p, v.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    std::vector<float> get() const {
        std::vector<float> v(n);
        CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
        return v;
    }
};

void same(const char* field, const std::vector<float>& a, const std::vector<float>& b,
          int nt, int pattern, bool snapshots, int step) {
    SQ_CHECK(a.size() == b.size(), "fixture comparison extent");
    for (size_t i = 0; i < a.size(); ++i) {
        SQ_CHECK(std::isfinite(a[i]) && std::isfinite(b[i]) &&
                 std::memcmp(&a[i], &b[i], sizeof(float)) == 0,
                 "%s NT%d pattern%d snapshots%d step%d index%zu old=%.9g prepared=%.9g",
                 field, nt, pattern, snapshots, step, i, a[i], b[i]);
        ++checks;
    }
}
void sentinel(const std::vector<float>& v, size_t first, size_t last, const char* field) {
    for (size_t i = first; i < last; ++i) {
        SQ_CHECK(std::memcmp(&v[i], &Sentinel, sizeof(float)) == 0, "%s guard overwritten index%zu", field, i);
        ++checks;
    }
}

void run(cudaStream_t stream, int nt, int pattern, bool snapshots) {
    const size_t stride = State + 17; // nontrivial snapshot pitch catches hidden contiguous assumptions
    const size_t snap_size = 2 * Guard + (sq::kMaxT - 1) * stride;
    const size_t out_size = 2 * Guard + sq::kMaxT * Out;
    const size_t factor_size = 2 * Guard + sq::gdn_prepare_floats(sq::kMaxT);
    Buffer conv(sq::kMaxT * Conv), ba(sq::kMaxT * 64), dt(Heads), ssm(Heads);
    Buffer state_a(State + 2 * Guard), state_b(State + 2 * Guard);
    Buffer out_a(out_size), out_b(out_size), snap_a(snap_size), snap_b(snap_size), factors(factor_size);
    std::vector<float> hc(conv.n, 0.f), hb(ba.n, 0.f), hd(Heads), ha(Heads);
    std::vector<float> initial(State + 2 * Guard, Sentinel), output(out_size, Sentinel);
    std::vector<float> snap(snap_size, Sentinel), factor(factor_size, Sentinel);
    std::mt19937 rng(610080 + pattern * 97 + nt);
    auto random = [&] { return ((int)(rng() % 20001) - 10000) / 10000.f; };
    for (int h = 0; h < Heads; ++h) {
        hd[h] = h % 3 == 0 ? 0.f : h % 3 == 1 ? .5f : -.5f;
        ha[h] = pattern == 5 ? (h % 4 == 0 ? -0.f : h % 4 == 1 ? 0.f : h % 4 == 2 ? .05f : -2.f)
                            : -.01f - .2f * std::abs(random());
    }
    for (size_t i = 0; i < State; ++i) {
        float v = random() * .1f;
        if (pattern == 1) v = i & 1 ? -0.f : 0.f;
        if (pattern == 3) v *= .001f;
        if (pattern == 5) v = i & 1 ? -.7f : .7f;
        initial[Guard + i] = v;
    }
    state_a.put(initial); state_b.put(initial); dt.put(hd); ssm.put(ha);
    for (int step = 0; step < 2; ++step) {
        // Reuse the exact same scratch after a different input window, also checking final state chaining.
        for (int t = 0; t < sq::kMaxT; ++t) {
            for (int i = 0; i < Conv; ++i) {
                float v = random();
                if (i < 4096) {
                    if (pattern == 1) v = i & 1 ? -0.f : 0.f;
                    if (pattern == 2) v *= 1.e-20f;
                    if (pattern == 3) v *= 10000.f;
                    if (pattern == 5) v = (i & 2 ? -.99f : .99f) + (i & 1 ? -.001f : .001f);
                } else if (pattern == 2 || pattern == 3) v *= .001f;
                hc[t * Conv + i] = v;
            }
            for (int h = 0; h < Heads; ++h) {
                const float beta_edges[] = {-90.f,-20.f,-0.f,0.f,20.f,90.f};
                const float alpha_edges[] = {-80.f,-20.f,-0.f,0.f,
                    std::nextafter(20.f, 0.f),20.f,std::nextafter(20.f, 30.f),40.f};
                hb[t * 64 + h] = pattern >= 4 ? beta_edges[(h + t + step) % 6] : random() * 2.f;
                hb[t * 64 + Heads + h] = pattern >= 4
                    ? alpha_edges[(h + t + step) % 8] - hd[h] : random() * 4.f;
            }
        }
        conv.put(hc); ba.put(hb); out_a.put(output); out_b.put(output);
        snap_a.put(snap); snap_b.put(snap); factors.put(factor);
        // cudaMemcpy above is on the default stream. Explicitly complete it before
        // this nonblocking stream consumes the fixture, avoiding a false parity race.
        CUDA_CHECK(cudaDeviceSynchronize());
        sq::k_gdn_recur(conv.p, ba.p, dt.p, ssm.p, state_a.p + Guard, out_a.p + Guard, nt,
                        snapshots ? snap_a.p + Guard : nullptr, stride, stream);
        sq::k_gdn_recur(conv.p, ba.p, dt.p, ssm.p, state_b.p + Guard, out_b.p + Guard, nt,
                        snapshots ? snap_b.p + Guard : nullptr, stride, stream, factors.p + Guard);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto state_old = state_a.get(), state_new = state_b.get();
        const auto output_old = out_a.get(), output_new = out_b.get();
        const auto snapshot_old = snap_a.get(), snapshot_new = snap_b.get();
        const auto factor_new = factors.get();
        same("state", state_old, state_new, nt, pattern, snapshots, step);
        same("output", output_old, output_new, nt, pattern, snapshots, step);
        same("snapshot", snapshot_old, snapshot_new, nt, pattern, snapshots, step);
        sentinel(state_new, 0, Guard, "state prefix");
        sentinel(state_new, Guard + State, state_new.size(), "state suffix");
        sentinel(output_new, 0, Guard, "output prefix");
        sentinel(output_new, Guard + nt * Out, output_new.size(), "output suffix");
        sentinel(factor_new, 0, Guard, "factor prefix");
        sentinel(factor_new, Guard + sq::gdn_prepare_floats(nt), factor_new.size(), "factor suffix");
        for (size_t i = Guard; i < Guard + sq::gdn_prepare_floats(nt); ++i) {
            SQ_CHECK(std::isfinite(factor_new[i]) && factor_new[i] != Sentinel, "factor unwritten or nonfinite");
            ++checks;
        }
        sentinel(snapshot_new, 0, Guard, "snapshot prefix");
        for (int t = 0; t < sq::kMaxT - 1; ++t) {
            const size_t first = Guard + t * stride;
            sentinel(snapshot_new, first + (snapshots && t + 1 < nt ? State : 0), first + stride, "snapshot pitch/tail");
        }
        sentinel(snapshot_new, snap_size - Guard, snap_size, "snapshot suffix");
        ++cases;
    }
}
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int nt = 1; nt <= sq::kMaxT; ++nt)
        for (int pattern = 0; pattern < 6; ++pattern)
            for (bool snapshots : {false, true}) run(stream, nt, pattern, snapshots);
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("GDN PREPARE EXACT PASS cases=%llu float_and_guard_checks=%llu NT1..4 random/zero/tiny/large/signed/softplus-edges chained-state/snapshots\n",
                (unsigned long long)cases, (unsigned long long)checks);
    return 0;
}
