// Standalone attention-level validation for the optional INT8 KV cache.
// Links sq_core. Does not load a model or write a cache/profile/configuration.
#include "kernels.cuh"
#include "quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {
constexpr int HD = 256, NH = 16, NK = 2, NS = 64, CTX = 25088, MT = 4;
constexpr int QKV = NH * 2 * HD + 2 * NK * HD;

void check(cudaError_t code, const char* what) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(code));
        std::exit(2);
    }
}
template <typename T> struct Device {
    T* p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { check(cudaMalloc(&p, count * sizeof(T)), "allocate"); }
    ~Device() { if (p) cudaFree(p); }
    void set(const T* source, size_t count) {
        check(cudaMemcpy(p, source, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<T> get(size_t count) const {
        std::vector<T> result(count);
        check(cudaMemcpy(result.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return result;
    }
};

std::vector<double> merge(const std::vector<float>& po, const std::vector<float>& pm, int pos, int T) {
    std::vector<double> out((size_t)T * NH * HD);
    for (int t = 0; t < T; ++t) {
        const int length = pos + t + 1;
        const int chunk = (((length + NS - 1) / NS) + 63) / 64 * 64;
        const int splits = (length + chunk - 1) / chunk;
        for (int h = 0; h < NH; ++h) {
            const size_t base = ((size_t)t * NH + h) * NS;
            double maximum = -std::numeric_limits<double>::infinity();
            for (int s = 0; s < splits; ++s) maximum = std::max(maximum, (double)pm[(base + s) * 2]);
            double denominator = 0;
            for (int s = 0; s < splits; ++s)
                denominator += std::exp((double)pm[(base + s) * 2] - maximum) * pm[(base + s) * 2 + 1];
            for (int d = 0; d < HD; ++d) {
                double numerator = 0;
                for (int s = 0; s < splits; ++s)
                    numerator += std::exp((double)pm[(base + s) * 2] - maximum) * po[(base + s) * HD + d];
                out[((size_t)t * NH + h) * HD + d] = numerator / denominator;
            }
        }
    }
    return out;
}

template <typename Fn> float elapsed(Fn fn, cudaStream_t stream) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    fn();
    check(cudaEventRecord(start, stream), "record start");
    constexpr int repetitions = 8;
    for (int i = 0; i < repetitions; ++i) fn();
    check(cudaEventRecord(stop, stream), "record stop");
    check(cudaEventSynchronize(stop), "wait event");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / repetitions;
}
}  // namespace

int main() {
    static_assert(MT == sq::kMaxT, "test shape must match engine");
    check(cudaSetDevice(0), "select device");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    const size_t rows = (size_t)NK * CTX, elements = rows * HD;
    std::vector<uint16_t> k16(elements), v16(elements);
    std::vector<int8_t> k8(elements), v8(elements);
    std::vector<float> kd(rows), vd(rows);
    std::mt19937 rng(0x1b8ca);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (size_t r = 0; r < rows; ++r) {
        float keys[HD], values[HD], km = 0, vm = 0;
        for (int d = 0; d < HD; ++d) {
            // Different key magnitudes create both diffuse and sharper softmax
            // distributions; occasional outliers stress the per-head scale.
            float k = normal(rng) * (0.2f + (r % 13) * 0.12f);
            float v = normal(rng) * (0.5f + (r % 3) * 0.5f);
            if (r % 97 == 0 && d == 37) { k *= 4; v *= 4; }
            k16[r * HD + d] = sq::f32_to_fp16(k);
            v16[r * HD + d] = sq::f32_to_fp16(v);
            keys[d] = sq::fp16_to_f32(k16[r * HD + d]);
            values[d] = sq::fp16_to_f32(v16[r * HD + d]);
            km = std::max(km, std::abs(keys[d]));
            vm = std::max(vm, std::abs(values[d]));
        }
        kd[r] = km / 127;
        vd[r] = vm / 127;
        for (int d = 0; d < HD; ++d) {
            k8[r * HD + d] = (int8_t)std::nearbyint(keys[d] / kd[r]);
            v8[r * HD + d] = (int8_t)std::nearbyint(values[d] / vd[r]);
        }
    }
    Device<uint16_t> dk16(elements), dv16(elements);
    Device<int8_t> dk8(elements), dv8(elements);
    Device<float> dkd(rows), dvd(rows);
    dk16.set(k16.data(), elements); dv16.set(v16.data(), elements);
    dk8.set(k8.data(), elements); dv8.set(v8.data(), elements);
    dkd.set(kd.data(), rows); dvd.set(vd.data(), rows);
    Device<float> qkv(MT * QKV), qnorm(HD), knorm(HD), q16(MT * NH * HD), q8(MT * NH * HD);
    Device<float> po16(MT * NH * NS * HD), po8(MT * NH * NS * HD), pm16(MT * NH * NS * 2), pm8(MT * NH * NS * 2);
    Device<sq::StepParams> params(1);
    std::vector<float> hqkv(MT * QKV), hnorm(HD);
    for (float& v : hnorm) v = 1.0f + normal(rng) * 0.15f;
    qnorm.set(hnorm.data(), HD);
    for (float& v : hnorm) v = 1.0f + normal(rng) * 0.15f;
    knorm.set(hnorm.data(), HD);
    size_t failures = 0, cases = 0;
    double worst_relative = 0, worst_absolute = 0;
    for (int length : {4, 65, 513, 4096, CTX}) {
        for (int T = 1; T <= MT; ++T) {
            const int pos = length - T;
            for (float& v : hqkv) v = normal(rng);
            // One zero value head exercises the d=0 representation as well.
            if (length == 4 && T == 1)
                std::fill(hqkv.begin() + NH * 2 * HD + NK * HD, hqkv.begin() + NH * 2 * HD + (NK + 1) * HD, 0.0f);
            qkv.set(hqkv.data(), hqkv.size());
            sq::StepParams hp{};
            hp.pos = pos; hp.n_tok = T; hp.seq = (uint32_t)(++cases);
            params.set(&hp, 1);
            sq::k_attn_prep(qkv.p, qnorm.p, knorm.p, 1e-6f, params.p, 1e7f, q16.p,
                            dk16.p, dv16.p, CTX, T, stream);
            sq::k_attn_prep_i8(qkv.p, qnorm.p, knorm.p, 1e-6f, params.p, 1e7f, q8.p,
                               dk8.p, dv8.p, dkd.p, dvd.p, CTX, T, stream);
            check(cudaStreamSynchronize(stream), "prepare caches");
            const auto aquery = q16.get((size_t)T * NH * HD), bquery = q8.get((size_t)T * NH * HD);
            if (std::memcmp(aquery.data(), bquery.data(), aquery.size() * sizeof(float))) {
                std::fprintf(stderr, "L%d T%d: query preparation differs from FP16 path\n", length, T);
                ++failures;
            }
            for (int h = 0; h < NK; ++h) for (int t = 0; t < T; ++t) {
                const size_t row = (size_t)h * CTX + pos + t;
                uint16_t fk[HD], fv[HD];
                int8_t ik[HD], iv[HD];
                float sk, sv;
                check(cudaMemcpy(fk, dk16.p + row * HD, sizeof(fk), cudaMemcpyDeviceToHost), "prepared K16");
                check(cudaMemcpy(fv, dv16.p + row * HD, sizeof(fv), cudaMemcpyDeviceToHost), "prepared V16");
                check(cudaMemcpy(ik, dk8.p + row * HD, sizeof(ik), cudaMemcpyDeviceToHost), "prepared K8");
                check(cudaMemcpy(iv, dv8.p + row * HD, sizeof(iv), cudaMemcpyDeviceToHost), "prepared V8");
                check(cudaMemcpy(&sk, dkd.p + row, sizeof(sk), cudaMemcpyDeviceToHost), "prepared K scale");
                check(cudaMemcpy(&sv, dvd.p + row, sizeof(sv), cudaMemcpyDeviceToHost), "prepared V scale");
                for (int d = 0; d < HD; ++d) {
                    const float ek = sq::fp16_to_f32(fk[d]), ev = sq::fp16_to_f32(fv[d]);
                    if (!std::isfinite(sk) || !std::isfinite(sv) || sk < 0 || sv < 0 ||
                        std::abs(sk * ik[d] - ek) > 0.501f * sk + 0.0006f * std::abs(ek) + 1e-6f ||
                        std::abs(sv * iv[d] - ev) > 0.501f * sv + 0.0006f * std::abs(ev) + 1e-6f) {
                        std::fprintf(stderr, "L%d T%d head%d token%d: KV quantization error exceeds rounding bound\n",
                                     length, T, h, t);
                        ++failures;
                        break;
                    }
                }
            }
            auto fp16 = [&] { sq::k_attn_decode(q16.p, dk16.p, dv16.p, params.p, CTX, po16.p, pm16.p, T, stream); };
            auto int8 = [&] { sq::k_attn_decode_i8(q8.p, dk8.p, dv8.p, dkd.p, dvd.p, params.p, CTX, po8.p, pm8.p, T, stream); };
            const float time16 = elapsed(fp16, stream), time8 = elapsed(int8, stream);
            const auto a = merge(po16.get((size_t)T * NH * NS * HD), pm16.get((size_t)T * NH * NS * 2), pos, T);
            const auto b = merge(po8.get((size_t)T * NH * NS * HD), pm8.get((size_t)T * NH * NS * 2), pos, T);
            double max_relative = 0, max_absolute = 0;
            for (int row = 0; row < T * NH; ++row) {
                double square_error = 0, square_reference = 0;
                for (int d = 0; d < HD; ++d) {
                    const size_t i = (size_t)row * HD + d;
                    const double error = a[i] - b[i];
                    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) ++failures;
                    square_error += error * error;
                    square_reference += a[i] * a[i];
                    max_absolute = std::max(max_absolute, std::abs(error));
                }
                const double relative = std::sqrt(square_error / std::max(square_reference, 1e-20));
                max_relative = std::max(max_relative, relative);
                if (relative > 0.035) ++failures;
            }
            worst_relative = std::max(worst_relative, max_relative);
            worst_absolute = std::max(worst_absolute, max_absolute);
            std::printf("KV L=%d T=%d: max head relativeL2 %.6f maxabs %.6g; FP16 %.3f ms INT8 %.3f ms\n",
                        length, T, max_relative, max_absolute, time16, time8);
        }
    }
    check(cudaStreamSynchronize(stream), "finish");
    cudaStreamDestroy(stream);
    std::printf("INT8 KV parity: %zu cases, %zu failures, worst head relativeL2 %.6f, worst absolute %.6g; "
                "criterion=finite outputs, identical queries, rounding bounds, relativeL2<=0.035\n",
                cases, failures, worst_relative, worst_absolute);
    std::printf("Per full-attention layer at25088: FP16 49.000MiB, INT8+FP32headscales24.8828125MiB; "
                "ten trunklayers save241.171875MiB. Approximate KV storage; full-model quality needs separate validation.\n");
    return failures ? 1 : 0;
}
