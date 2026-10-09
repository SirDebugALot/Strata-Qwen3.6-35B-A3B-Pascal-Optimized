// Standalone CUDA parity test for iq_native.cuh. This is NOT an engine source.
// Build an executable linked to sq_cpu and CUDA; optionally pass a GGUF path to
// cover actual model blocks as well as randomized blocks and signed edge inputs.
// See iq_native.cuh and third_party/ggml/LICENSE for source attribution.
#include "iq_native.cuh"
#include "gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(2);
    }
}

struct TestCase {
    uint32_t type;
    uint32_t offset;
    int8_t activations[4][256];
    float scales[4][8];
};

__global__ void test_dots(const uint8_t* blobs, const TestCase* cases, float* results, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * 8) return;
    const TestCase& c = cases[index / 8];
    const int group = index % 8;
    const uint8_t* block = blobs + c.offset;
    const int8_t* q = c.activations[0] + 32 * group;
    const float d = c.scales[0][group];
    switch (c.type) {
        case sq::T_Q2_K: results[index] = sq::iq_native::dot_32<sq::T_Q2_K>(block, group, q, d); break;
        case sq::T_Q3_K: results[index] = sq::iq_native::dot_32<sq::T_Q3_K>(block, group, q, d); break;
        case sq::T_IQ2_S: results[index] = sq::iq_native::dot_32<sq::T_IQ2_S>(block, group, q, d); break;
        case sq::T_IQ3_S: results[index] = sq::iq_native::dot_32<sq::T_IQ3_S>(block, group, q, d); break;
        case sq::T_IQ4_XS: results[index] = sq::iq_native::dot_32<sq::T_IQ4_XS>(block, group, q, d); break;
    }
}

template <uint32_t Type, int NT>
__device__ void test_multi_type(const uint8_t* block, const TestCase& c, int group, unsigned active, float* result) {
    const int8_t* q[NT];
    float d[NT], actual[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        q[t] = active & (1u << t) ? c.activations[t] + 32 * group : nullptr;
        d[t] = c.scales[t][group];
    }
    sq::iq_native::dot_32_multi<Type, NT>(block, group, q, d, active, actual);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        result[2 * t] = actual[t];
        result[2 * t + 1] = active & (1u << t) ? sq::iq_native::dot_32<Type>(block, group, q[t], d[t]) : 0.0f;
    }
}

template <int NT>
__global__ void test_dots_multi(const uint8_t* blobs, const TestCase* cases, float* results, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * 8) return;
    const TestCase& c = cases[index / 8];
    const int group = index % 8;
    const unsigned active = blockIdx.y;
    const uint8_t* block = blobs + c.offset;
    float* out = results + ((size_t)active * n * 8 + index) * NT * 2;
    switch (c.type) {
        case sq::T_Q2_K: test_multi_type<sq::T_Q2_K, NT>(block, c, group, active, out); break;
        case sq::T_Q3_K: test_multi_type<sq::T_Q3_K, NT>(block, c, group, active, out); break;
        case sq::T_IQ2_S: test_multi_type<sq::T_IQ2_S, NT>(block, c, group, active, out); break;
        case sq::T_IQ3_S: test_multi_type<sq::T_IQ3_S, NT>(block, c, group, active, out); break;
        case sq::T_IQ4_XS: test_multi_type<sq::T_IQ4_XS, NT>(block, c, group, active, out); break;
    }
}

bool native_type(uint32_t type) {
    return type == sq::T_IQ2_S || type == sq::T_IQ3_S || type == sq::T_IQ4_XS ||
           type == sq::T_Q2_K || type == sq::T_Q3_K;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2) {
        std::fprintf(stderr, "usage: test_iq_native [model.gguf]\n");
        return 2;
    }
    std::mt19937 rng(0x6a31);
    std::uniform_int_distribution<int> quant(-127, 127), byte(0, 255);
    std::vector<uint8_t> blobs;
    std::vector<TestCase> cases;
    size_t real_blocks = 0;
    auto append = [&](uint32_t type, const uint8_t* data, int mode) {
        const size_t bytes = static_cast<size_t>(sq::row_bytes(type, 256));
        // Exercise both two-byte alignments used by consecutive 82/110-byte IQ
        // blocks. IQ4_XS has a four-byte-aligned packed-nibble field.
        const size_t alignment = type == sq::T_IQ4_XS || type == sq::T_Q2_K ? 4 : 2;
        while (blobs.size() % alignment) blobs.push_back(0);
        TestCase c{};
        c.type = type;
        c.offset = static_cast<uint32_t>(blobs.size());
        for (int t = 0; t < 4; ++t) {
            const int token_mode = (mode + t) % 5;
            for (int j = 0; j < 256; ++j) {
                if (token_mode == 0) c.activations[t][j] = static_cast<int8_t>(quant(rng));
                else if (token_mode == 1) c.activations[t][j] = -127;
                else if (token_mode == 2) c.activations[t][j] = 127;
                else if (token_mode == 3) c.activations[t][j] = (j & 1) ? 127 : -127;
                else c.activations[t][j] = 0;
            }
            for (int j = 0; j < 8; ++j)
                c.scales[t][j] = ((cases.size() + t) % 17 == 0) ? 0.0f : (1.0f + static_cast<float>(rng() % 127)) / 1024.0f;
        }
        blobs.insert(blobs.end(), data, data + bytes);
        cases.push_back(c);
    };
    const float block_scales[] = {0.0f, 1.0f / 4096, 1.0f / 256, 0.1f, 1.0f, 2.0f, -0.25f};
    for (uint32_t type : {sq::T_IQ2_S, sq::T_IQ3_S, sq::T_IQ4_XS, sq::T_Q2_K, sq::T_Q3_K}) {
        const int bytes = static_cast<int>(sq::row_bytes(type, 256));
        for (int i = 0; i < 320; ++i) {
            std::array<uint8_t, 136> data{};
            for (int j = 0; j < bytes; ++j) data[j] = static_cast<uint8_t>(byte(rng));
            const uint16_t d = sq::f32_to_fp16(block_scales[i % 7]);
            const int scale_offset = type == sq::T_Q2_K ? 80 : type == sq::T_Q3_K ? 108 : 0;
            std::memcpy(data.data() + scale_offset, &d, sizeof(d));
            if (type == sq::T_Q2_K) {
                const uint16_t dm = sq::f32_to_fp16(block_scales[(i / 7) % 7]);
                std::memcpy(data.data() + 82, &dm, sizeof(dm));
            }
            append(type, data.data(), i % 5);
        }
    }
    if (argc == 2) {
        sq::GgufFile model;
        std::string error;
        if (!model.open(argv[1], error)) {
            std::fprintf(stderr, "GGUF: %s\n", error.c_str());
            return 2;
        }
        for (const auto& tensor : model.tensors()) {
            if (!native_type(tensor.type) || tensor.name.find("_exps.") == std::string::npos) continue;
            const int64_t nb = tensor.n_elements() / 256;
            const int64_t bytes = sq::row_bytes(tensor.type, 256);
            const int64_t locations[] = {0, 1, nb / 2, nb - 1};
            for (int64_t index : locations) {
                if (index < 0 || index >= nb) continue;
                append(tensor.type, tensor.data + index * bytes, 0);
                ++real_blocks;
            }
        }
    }
    const int n = static_cast<int>(cases.size());
    uint8_t* device_blobs = nullptr;
    TestCase* device_cases = nullptr;
    float* device_results = nullptr;
    check(cudaMalloc(&device_blobs, blobs.size()), "allocate blocks");
    check(cudaMalloc(&device_cases, cases.size() * sizeof(TestCase)), "allocate cases");
    check(cudaMalloc(&device_results, cases.size() * 8 * sizeof(float)), "allocate results");
    check(cudaMemcpy(device_blobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice), "upload blocks");
    check(cudaMemcpy(device_cases, cases.data(), cases.size() * sizeof(TestCase), cudaMemcpyHostToDevice), "upload cases");
    test_dots<<<(n * 8 + 127) / 128, 128>>>(device_blobs, device_cases, device_results, n);
    check(cudaGetLastError(), "launch native IQ test");
    std::vector<float> results(cases.size() * 8);
    check(cudaMemcpy(results.data(), device_results, results.size() * sizeof(float), cudaMemcpyDeviceToHost), "read results");
    check(cudaFree(device_results), "free results");
    size_t failures = 0;
    double max_absolute = 0, max_normalized = 0;
    size_t groups_by_type[5] = {};
    for (size_t i = 0; i < cases.size(); ++i) {
        const TestCase& c = cases[i];
        float weights[256];
        sq::dequant_row(c.type, blobs.data() + c.offset, weights, 256);
        for (int group = 0; group < 8; ++group) {
            double expected = 0, magnitude = 0;
            for (int j = 32 * group; j < 32 * (group + 1); ++j) {
                const double value = static_cast<double>(weights[j]) * c.activations[0][j] * c.scales[0][group];
                expected += value;
                magnitude += std::abs(value);
            }
            const double actual = results[i * 8 + group];
            const double delta = std::abs(actual - expected);
            const double tolerance = 1e-6 + 2e-6 * magnitude;
            max_absolute = std::max(max_absolute, delta);
            max_normalized = std::max(max_normalized, delta / std::max(magnitude, 1e-20));
            ++groups_by_type[c.type == sq::T_IQ2_S ? 0 : c.type == sq::T_IQ3_S ? 1 :
                             c.type == sq::T_IQ4_XS ? 2 : c.type == sq::T_Q2_K ? 3 : 4];
            if (!std::isfinite(actual) || delta > tolerance) {
                if (failures < 12)
                    std::fprintf(stderr, "%s case %zu group %d: actual %.9g expected %.9g delta %.6g tolerance %.6g\n",
                                 sq::type_name(c.type), i, group, actual, expected, delta, tolerance);
                ++failures;
            }
        }
    }
    std::printf("native IQ/K GPU parity: %zu blocks (%zu real GGUF), %zu groups, %zu failures; max absolute %.9g, "
                "max error / sum(abs(products)) %.9g\n",
                cases.size(), real_blocks, cases.size() * 8, failures, max_absolute, max_normalized);
    std::printf("groups: IQ2_S %zu, IQ3_S %zu, IQ4_XS %zu, Q2_K %zu, Q3_K %zu; "
                "reference=original GGML dequantized weights x unchanged ActQ\n",
                groups_by_type[0], groups_by_type[1], groups_by_type[2], groups_by_type[3], groups_by_type[4]);

    // Every active-token mask, including empty/single/sparse, must match repeated scalar helpers
    // bitwise. Inactive pointers are null, so a missing route must not issue an activation load.
    size_t batch_comparisons = 0, batch_failures = 0;
    check(cudaMalloc(&device_results, (size_t)n * 8 * 16 * 4 * 2 * sizeof(float)), "allocate batch results");
    for (int nt = 2; nt <= 4; ++nt) {
        const dim3 grid((n * 8 + 127) / 128, 1u << nt);
        if (nt == 2) test_dots_multi<2><<<grid, 128>>>(device_blobs, device_cases, device_results, n);
        if (nt == 3) test_dots_multi<3><<<grid, 128>>>(device_blobs, device_cases, device_results, n);
        if (nt == 4) test_dots_multi<4><<<grid, 128>>>(device_blobs, device_cases, device_results, n);
        check(cudaGetLastError(), "launch native expert reuse test");
        std::vector<float> batch((size_t)n * 8 * (1u << nt) * nt * 2);
        check(cudaMemcpy(batch.data(), device_results, batch.size() * sizeof(float), cudaMemcpyDeviceToHost), "read batch results");
        for (size_t i = 0; i < batch.size(); i += 2) {
            ++batch_comparisons;
            if (!std::isfinite(batch[i]) || std::memcmp(&batch[i], &batch[i + 1], sizeof(float))) {
                if (batch_failures < 12) {
                    const size_t pair = i / 2, group_case = (pair / nt) % ((size_t)n * 8);
                    std::fprintf(stderr, "native reuse %s NT=%d mask=%zu case=%zu group=%zu token=%zu: actual %.9g T1 %.9g\n",
                                 sq::type_name(cases[group_case / 8].type), nt, pair / (nt * (size_t)n * 8),
                                 group_case / 8, group_case % 8, pair % nt, batch[i], batch[i + 1]);
                }
                ++batch_failures;
            }
        }
    }
    check(cudaFree(device_results), "free batch results");
    check(cudaFree(device_cases), "free cases");
    check(cudaFree(device_blobs), "free blocks");
    std::printf("native IQ/K reuse parity: NT=2..4 all active masks, %zu bitwise comparisons, %zu failures\n",
                batch_comparisons, batch_failures);
    failures += batch_failures;
    return failures ? 1 : 0;
}
