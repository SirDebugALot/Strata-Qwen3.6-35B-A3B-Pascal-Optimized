// GPU unit probe: original Q6 dequantized weights x exactly the same ActQ input. Optional GGUF argument reads
// only67 real rows per shape in addition to random blocks. Parent serializes GPU runs; no model cache mutation.
#include "dense_q6.cuh"
#include "dense_q6_layout.hpp"
#include "common.hpp"
#include "gguf.hpp"
#include "quant.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace {
template <typename T> T* allocate(size_t count) {
    T* result = nullptr;
    CUDA_CHECK(cudaMalloc(&result, count * sizeof(T)));
    return result;
}
template <typename T> void upload(T* destination, const std::vector<T>& source) {
    CUDA_CHECK(cudaMemcpy(destination, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

__global__ void probe_centering_kernel(uint32_t* failures) {
    const uint32_t combination = blockIdx.x * blockDim.x + threadIdx.x;
    if (combination >= (1u << 24)) return;
    uint32_t packed = 0, reference = 0;
#pragma unroll
    for (int byte = 0; byte < 4; ++byte) {
        const int value = (combination >> (byte * 6)) & 63;
        packed |= (uint32_t)value << (byte * 8);
        reference |= ((uint32_t)(value - 32) & 255u) << (byte * 8);
    }
    if ((uint32_t)sq::detail::dense_q6_center<false>(packed) != reference ||
        (uint32_t)sq::detail::dense_q6_center<true>(packed) != reference) {
        atomicAdd(failures, 1u);
        atomicMin(failures + 1, combination);
    }
}

void probe_centering() {
    // Exhaust all 64^4 packed operand combinations, not just repeated-byte cases. Only two device words
    // are retained; no tensor fixture, quantization, or floating-point tolerance enters this assertion.
    uint32_t* failures = allocate<uint32_t>(2);
    upload(failures, std::vector<uint32_t>{0u, 0xffffffffu});
    probe_centering_kernel<<<(1u << 24) / 256, 256>>>(failures);
    CUDA_CHECK(cudaGetLastError());
    uint32_t result[2];
    CUDA_CHECK(cudaMemcpy(result, failures, sizeof(result), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(failures));
    SQ_CHECK(result[0] == 0, "Q6 packed centering failed %u combinations; first=0x%06x", result[0], result[1]);
    sq::log("Q6 packed centering: all 16777216 four-byte combinations match scalar signed-byte reference exactly");
}

void probe(const std::vector<uint8_t>& packed, int cols, const char* fixture) {
    constexpr int rows = 67, maximum_tokens = 4;
    const size_t blocks = (size_t)cols / 256, source_row = blocks * 210;
    const size_t plane_row = sq::dense_q6_plane_row_bytes(cols);
    SQ_CHECK(plane_row % 32 == 0 && plane_row >= source_row && plane_row - source_row < 32,
             "invalid Q6 plane row padding");
    std::vector<uint8_t> planes(rows * plane_row, 0xcd);
    for (int row = 0; row < rows; ++row) {
        const uint8_t* original = packed.data() + row * source_row;
        uint8_t* reordered = planes.data() + row * plane_row;
        sq::dense_q6_pack_row_planes(original, reordered, cols);
        // Check every original byte independently, including both unaligned 210-byte block parities.
        // The GPU later compares the full arithmetic against the unrelated original-block/dequant paths.
        for (size_t block = 0; block < blocks; ++block) {
            for (size_t byte = 0; byte < 128; ++byte) {
                const size_t segment = byte / 32, word = byte % 32 / 4, within = byte % 4;
                SQ_CHECK(reordered[4 * (word * blocks * 4 + block * 4 + segment) + within] == original[block * 210 + byte],
                         "Q6 low-plane byte changed: %s cols%d row%d block%zu byte%zu", fixture, cols, row, block, byte);
            }
            for (size_t byte = 0; byte < 64; ++byte) {
                const size_t half = byte / 32, word = byte % 32 / 4, within = byte % 4;
                SQ_CHECK(reordered[128 * blocks + 4 * (word * blocks * 2 + block * 2 + half) + within] ==
                         original[block * 210 + 128 + byte], "Q6 high-plane byte changed");
            }
            SQ_CHECK(std::memcmp(reordered + blocks * 192 + block * 16, original + block * 210 + 192, 16) == 0 &&
                     std::memcmp(reordered + blocks * 208 + block * 2, original + block * 210 + 208, 2) == 0,
                     "Q6 repack changed scale bytes");
        }
        for (size_t byte = source_row; byte < plane_row; ++byte)
            SQ_CHECK(reordered[byte] == 0, "Q6 row padding was not initialized");
    }
    std::mt19937 rng(61030 + cols);
    std::vector<int8_t> aq(maximum_tokens * cols);
    std::vector<float> ad(aq.size() / 32), sums(ad.size(), 0);
    for (auto& v : aq) v = (int8_t)((int)(rng() % 256) - 128);
    for (size_t g = 0; g < ad.size(); ++g) {
        ad[g] = (float)(1 + rng() % 7) / 4096;
        for (int j = 0; j < 32; ++j) sums[g] += aq[g * 32 + j];
        sums[g] *= ad[g];
    }
    sq::DQ8 weight;
    weight.rows = rows;
    weight.cols = cols;
    uint8_t* device_weights = allocate<uint8_t>(packed.size());
    weight.q6 = device_weights;
    upload(device_weights, packed);
    uint8_t* device_planes = allocate<uint8_t>(planes.size());
    upload(device_planes, planes);
    sq::DQ8 repacked_weight = weight;
    repacked_weight.q6 = device_planes;
    repacked_weight.q6_row_planes = true;
    sq::ActQ activation{allocate<int8_t>(aq.size()), allocate<float>(ad.size()), allocate<float>(sums.size())};
    upload(activation.q, aq); upload(activation.d, ad); upload(activation.s, sums);
    float* output = allocate<float>(rows * maximum_tokens);
    float* single = allocate<float>(rows * maximum_tokens);
    float* transposed = allocate<float>(rows * maximum_tokens);
    std::vector<float> dequant((size_t)rows * cols);
    for (int row = 0; row < rows; ++row)
        sq::dequant_row(sq::T_Q6_K, packed.data() + (size_t)row * cols / 256 * 210,
                       dequant.data() + (size_t)row * cols, cols);
    std::vector<float> host_output(rows * maximum_tokens), host_single(host_output.size()), host_transposed(host_output.size());
    for (int tokens = 1; tokens <= maximum_tokens; ++tokens) {
        // Test the public dense dispatch too, including its optional descriptor tag.
        sq::k_gemv_q8(weight, activation, output, tokens, nullptr);
        sq::k_test_gemv_q6(weight, activation, single, tokens, false, nullptr);
        sq::k_test_gemv_q6(weight, activation, transposed, tokens, true, nullptr);
        CUDA_CHECK(cudaGetLastError());
        const size_t result_bytes = (size_t)tokens * rows * sizeof(float);
        CUDA_CHECK(cudaMemcpy(host_output.data(), output, result_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_single.data(), single, result_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_transposed.data(), transposed, result_bytes, cudaMemcpyDeviceToHost));
        SQ_CHECK(std::memcmp(host_output.data(), host_single.data(), result_bytes) == 0 &&
                 std::memcmp(host_output.data(), host_transposed.data(), result_bytes) == 0,
                 "Q6 original/transposed/public dispatch differ: %s cols%d T%d", fixture, cols, tokens);
        for (int layout = 0; layout < 3; ++layout) {
            if (layout == 2) sq::k_gemv_q8(repacked_weight, activation, transposed, tokens, nullptr);
            else sq::k_test_gemv_q6(repacked_weight, activation, transposed, tokens, layout != 0, nullptr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(host_transposed.data(), transposed, result_bytes, cudaMemcpyDeviceToHost));
            SQ_CHECK(std::memcmp(host_output.data(), host_transposed.data(), result_bytes) == 0,
                     "Q6 original/repacked storage differs: %s cols%d T%d activation_layout%d", fixture, cols, tokens, layout);
        }
        // Explicit fast-centering variants must be bitwise identical for each storage/shared-memory layout,
        // regardless of which variant the environment selected for the public dispatch above.
        for (int storage = 0; storage < 2; ++storage) for (int layout = 0; layout < 2; ++layout) {
            const sq::DQ8& selected_weight = storage ? repacked_weight : weight;
            sq::k_test_gemv_q6(selected_weight, activation, transposed, tokens, layout != 0, nullptr, true);
            for (int t = 0; t < tokens; ++t)
                sq::k_test_gemv_q6(selected_weight, activation.row(t, cols), single + t * rows, 1,
                                   layout != 0, nullptr, true);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(host_transposed.data(), transposed, result_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_single.data(), single, result_bytes, cudaMemcpyDeviceToHost));
            SQ_CHECK(std::memcmp(host_output.data(), host_transposed.data(), result_bytes) == 0 &&
                     std::memcmp(host_output.data(), host_single.data(), result_bytes) == 0,
                     "Q6 fast centering differs from intrinsic/multiT: %s cols%d T%d storage%d layout%d",
                     fixture, cols, tokens, storage, layout);
        }
        for (int t = 0; t < tokens; ++t)
            sq::k_test_gemv_q6(weight, activation.row(t, cols), single + t * rows, 1, true, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_output.data(), output, (size_t)tokens * rows * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_single.data(), single, (size_t)tokens * rows * sizeof(float), cudaMemcpyDeviceToHost));
        SQ_CHECK(std::memcmp(host_output.data(), host_single.data(), (size_t)tokens * rows * sizeof(float)) == 0,
                 "dense Q6 multi-token differs from T1: %s cols%d T%d", fixture, cols, tokens);
        for (int t = 0; t < tokens; ++t)
            sq::k_test_gemv_q6(repacked_weight, activation.row(t, cols), single + t * rows, 1, true, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_single.data(), single, result_bytes, cudaMemcpyDeviceToHost));
        SQ_CHECK(std::memcmp(host_output.data(), host_single.data(), result_bytes) == 0,
                 "dense Q6 repacked multi-token differs from repeated T1: %s cols%d T%d", fixture, cols, tokens);
        double worst = 0, squared_error = 0, squared_reference = 0;
        for (int t = 0; t < tokens; ++t) for (int row = 0; row < rows; ++row) {
            double reference = 0;
            for (int i = 0; i < cols; ++i)
                reference += (double)dequant[(size_t)row * cols + i] * aq[t * cols + i] * ad[t * cols / 32 + i / 32];
            const float actual = host_output[t * rows + row];
            SQ_CHECK(std::isfinite(actual), "nonfinite native Q6 result");
            const double error = std::abs(actual - reference);
            worst = std::max(worst, error / (1 + std::abs(reference)));
            squared_error += error * error;
            squared_reference += reference * reference;
        }
        const double relative_l2 = std::sqrt(squared_error / std::max(1e-30, squared_reference));
        SQ_CHECK(worst < 2e-5 && relative_l2 < 1e-5,
                 "Q6 original-weight reference mismatch %s cols%d T%d scaled%.9g relativeL2%.9g",
                 fixture, cols, tokens, worst, relative_l2);
        sq::log("Q6 %s cols%d T%d original_weight_scaled=%.9g relativeL2=%.9g multiT_transpose_repack_centering_exact=PASS",
                fixture, cols, tokens, worst, relative_l2);
    }
    // Mixed valid IDs, repeated IDs and invalid rows. The same kernel/reduction must equal the full projection.
    const std::vector<int32_t> ids{66, 0, 13, 13, -1, 67, 1, 0, 65, 32, 64, 2, 3, 4, 5, 6, 7};
    std::vector<int32_t> first(rows, std::numeric_limits<int32_t>::max());
    for (int i = 0; i < (int)ids.size(); ++i)
        if (ids[i] >= 0 && ids[i] < rows) first[ids[i]] = std::min(first[ids[i]], i);
    int32_t* device_ids = allocate<int32_t>(ids.size());
    int32_t* device_first = allocate<int32_t>(first.size());
    float* gathered = allocate<float>(ids.size());
    upload(device_ids, ids); upload(device_first, first);
    std::vector<float> host_gathered(ids.size());
    // The deployed gathered head is NT1. Exercise each of the four activation rows against its NT4 full-head
    // result, for both storage/centering variants and both explicit shared layouts plus production dispatch.
    for (int storage = 0; storage < 2; ++storage) for (int token = 0; token < maximum_tokens; ++token)
    for (int layout = 0; layout < 5; ++layout) {
        const sq::DQ8& selected_weight = storage ? repacked_weight : weight;
        const sq::ActQ selected_activation = activation.row(token, cols);
        if (layout == 2) sq::k_gather_q6(selected_weight, selected_activation, device_ids, device_first, (int)ids.size(), gathered, nullptr);
        else sq::k_test_gather_q6(selected_weight, selected_activation, device_ids, device_first, (int)ids.size(),
                                    gathered, layout == 1 || layout == 4, nullptr, layout >= 3);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(host_gathered.data(), gathered, ids.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int slot = 0; slot < (int)ids.size(); ++slot) {
            const int id = ids[slot];
            if (id < 0 || id >= rows || first[id] != slot)
                SQ_CHECK(host_gathered[slot] == -INFINITY, "invalid/duplicate Q6 gathered row was not masked");
            else SQ_CHECK(std::memcmp(&host_gathered[slot], &host_output[token * rows + id], sizeof(float)) == 0,
                          "Q6 gather differs from full head: %s cols%d slot%d layout%d storage%d token%d",
                          fixture, cols, slot, layout, storage, token);
        }
    }
    sq::log("Q6 %s cols%d gather_storage_activation_centering_layouts_exact_and_invalid_mask=PASS", fixture, cols);
    CUDA_CHECK(cudaFree(device_ids)); CUDA_CHECK(cudaFree(device_first)); CUDA_CHECK(cudaFree(gathered));
    CUDA_CHECK(cudaFree(device_weights)); CUDA_CHECK(cudaFree(device_planes)); CUDA_CHECK(cudaFree(output)); CUDA_CHECK(cudaFree(single));
    CUDA_CHECK(cudaFree(transposed));
    CUDA_CHECK(cudaFree(activation.q)); CUDA_CHECK(cudaFree(activation.d)); CUDA_CHECK(cudaFree(activation.s));
}
} // namespace

int main(int argc, char** argv) {
    probe_centering();
    std::unique_ptr<sq::GgufFile> model;
    if (argc > 1) {
        model = std::make_unique<sq::GgufFile>();
        std::string error;
        SQ_CHECK(model->open(argv[1], error), "%s", error.c_str());
    }
    std::mt19937 rng(61030);
    for (int cols : {512, 2048, 4096}) {
        std::vector<uint8_t> packed((size_t)67 * cols / 256 * 210);
        for (auto& v : packed) v = (uint8_t)rng();
        for (size_t block = 0; block < packed.size(); block += 210) {
            const uint16_t d = sq::f32_to_fp16((float)(1 + rng() % 7) / 4096);
            std::memcpy(packed.data() + block + 208, &d, sizeof(d));
        }
        probe(packed, cols, "random");
        if (model) {
            const sq::GgufTensor* selected = nullptr;
            for (const auto& tensor : model->tensors())
                if (tensor.type == sq::T_Q6_K && tensor.ne[0] == (uint64_t)cols && tensor.bytes >= (int64_t)packed.size()) {
                    selected = &tensor; break;
                }
            SQ_CHECK(selected, "no real Q6 tensor for cols%d", cols);
            std::memcpy(packed.data(), selected->data, packed.size());
            probe(packed, cols, selected->name.c_str());
        }
    }
    sq::log("NATIVE DENSE Q6 ORIGINAL-WEIGHT / MULTI-TOKEN / GATHER / LOSSLESS ROW-PLANE REPACK / EXACT PACKED CENTERING PASS");
    return 0;
}
