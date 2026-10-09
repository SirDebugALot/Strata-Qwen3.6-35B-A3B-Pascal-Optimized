// CPU-only exact Q2_K/Q3_K shared-unpack regression. No GPU initialization or model-cache writes.
// Usage: test_cpu_k23_multitoken [model.gguf] -- real MTP gate/up/down rows are optional but recommended.
#include "cpu_iq.hpp"
#include "common.hpp"
#include "gguf.hpp"
#include "quant.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace {
constexpr int Rows = 19, Capacity = Rows + 3;
uint64_t cases = 0, checked = 0;

std::vector<uint8_t> random_weights(uint32_t type, int cols, std::mt19937& rng) {
    const size_t block_bytes = type == sq::T_Q2_K ? 84 : 110;
    std::vector<uint8_t> result((size_t)Rows * sq::row_bytes(type, cols));
    for (uint8_t& b : result) b = (uint8_t)rng();
    for (size_t b = 0; b < result.size(); b += block_bytes) {
        const uint16_t d = sq::f32_to_fp16((float)((int)(rng() % 9) - 4) / 4096);
        std::memcpy(result.data() + b + (type == sq::T_Q2_K ? 80 : 108), &d, 2);
        if (type == sq::T_Q2_K) {
            const uint16_t minimum = sq::f32_to_fp16((float)(rng() % 5) / 4096);
            std::memcpy(result.data() + b + 82, &minimum, 2);
        }
    }
    return result;
}

void probe(uint32_t type, int cols, const uint8_t* weights, const uint8_t* up, const char* fixture) {
    const size_t row_bytes = (size_t)sq::row_bytes(type, cols);
    const size_t up_offset = Rows * row_bytes + 18; // non-contiguous gate/up bases and 2-byte alignment are intentional
    std::vector<uint8_t> blob(up_offset + Rows * row_bytes);
    std::memcpy(blob.data(), weights, Rows * row_bytes);
    if (up) std::memcpy(blob.data() + up_offset, up, Rows * row_bytes);
    std::array<std::vector<sq::Q8K>, 4> activations;
    std::mt19937 rng(610230 + type + cols);
    for (int t = 0; t < 4; ++t) {
        activations[t].resize(cols / 256);
        for (sq::Q8K& block : activations[t]) {
            std::memset(&block, 0, sizeof(block));
            block.d = (float)(1 + rng() % 7) / 4096;
            for (int i = 0; i < 256; ++i) {
                block.nat[i] = t == 0 ? 0 : t == 2 ? (i & 1 ? -128 : 127) : (int8_t)((int)(rng() % 256) - 128);
                block.bsums[i / 16] += block.nat[i];
            }
        }
    }
    const int mappings[][4] = {{0, 1, 2, 3}, {3, 1, 0, 2}, {1, 1, 3, 1}};
    const auto dot = type == sq::T_Q2_K ? sq::native_iq::dot_q2k : sq::native_iq::dot_q3k;
    for (int nt = 1; nt <= 4; ++nt) for (const auto& mapping : mappings) for (int range = 0; range < 2; ++range) {
        const int r0 = range ? 3 : 0, r1 = range ? Rows - 2 : Rows;
        const void* acts[4];
        float *outputs[4], *gu_outputs[4];
        std::array<std::array<float, Capacity>, 4> actual, reference, actual_gu, reference_gu;
        for (int t = 0; t < 4; ++t) {
            acts[t] = activations[mapping[t]].data();
            outputs[t] = actual[t].data(); gu_outputs[t] = actual_gu[t].data();
            actual[t].fill(123456.25f); reference[t].fill(123456.25f);
            actual_gu[t].fill(123456.25f); reference_gu[t].fill(123456.25f);
            if (t >= nt) continue;
            for (int row = r0; row < r1; ++row) {
                const auto* activation = (const sq::Q8K*)acts[t];
                const float g = dot(blob.data() + row * row_bytes, activation, cols / 256);
                reference[t][row] = g;
                if (up) {
                    const float u = dot(blob.data() + up_offset + row * row_bytes, activation, cols / 256);
                    reference_gu[t][row] = (g / (1.f + std::exp(-g))) * u;
                }
            }
        }
        sq::native_iq::k23_rows(type, blob.data(), row_bytes, cols, acts, nt, outputs, r0, r1);
        if (up) sq::native_iq::k23_gu_rows(type, blob.data(), row_bytes, up_offset, cols, acts, nt, gu_outputs, r0, r1);
        for (int t = 0; t < 4; ++t) for (int row = 0; row < Capacity; ++row) {
            SQ_CHECK(std::isfinite(actual[t][row]) &&
                     std::memcmp(&reference[t][row], &actual[t][row], sizeof(float)) == 0,
                     "%s type%u cols%d NT%d mapping%d%d%d%d token%d row%d range%d dot differs: %.9g/%.9g",
                     fixture, type, cols, nt, mapping[0], mapping[1], mapping[2], mapping[3], t, row, range,
                     reference[t][row], actual[t][row]);
            ++checked;
            if (up) {
                SQ_CHECK(std::isfinite(actual_gu[t][row]) &&
                         std::memcmp(&reference_gu[t][row], &actual_gu[t][row], sizeof(float)) == 0,
                         "%s type%u cols%d NT%d token%d row%d range%d gate/up differs: %.9g/%.9g",
                         fixture, type, cols, nt, t, row, range, reference_gu[t][row], actual_gu[t][row]);
                ++checked;
            }
        }
        ++cases;
    }
    sq::log("Q2/Q3 shared rows: %s type%u cols%d T1..4/gapped/duplicate/range/guards%s bit-exact=PASS",
            fixture, type, cols, up ? "/gate-up" : "/down");
}
} // namespace

int main(int argc, char** argv) {
    SQ_CHECK(sq::cpu_has_avx2(), "test requires AVX2/F16C CPU");
    std::mt19937 rng(610230);
    for (uint32_t type : {sq::T_Q2_K, sq::T_Q3_K}) for (int cols : {512, 2048}) {
        const auto weights = random_weights(type, cols, rng), up = random_weights(type, cols, rng);
        probe(type, cols, weights.data(), up.data(), "random");
    }
    if (argc > 1) {
        sq::GgufFile gguf;
        std::string error;
        SQ_CHECK(gguf.open(argv[1], error), "%s", error.c_str());
        const auto& gate = gguf.need("blk.40.ffn_gate_exps.weight");
        const auto& up = gguf.need("blk.40.ffn_up_exps.weight");
        const auto& down = gguf.need("blk.40.ffn_down_exps.weight");
        SQ_CHECK(gate.type == sq::T_Q2_K && up.type == gate.type && down.type == sq::T_Q3_K &&
                 gate.ne[0] == 2048 && gate.ne[1] == 512 && gate.ne[2] == 256 &&
                 down.ne[0] == 512 && down.ne[1] == 2048 && down.ne[2] == 256,
                 "real fixture is not the expected Q2_K/Q3_K MTP block");
        const size_t gate_bytes = (size_t)sq::row_bytes(gate.type, 2048) * 512;
        const size_t down_bytes = (size_t)sq::row_bytes(down.type, 512) * 2048;
        for (int expert : {0, 17, 255}) {
            char fixture[48]; std::snprintf(fixture, sizeof(fixture), "real-MTP-expert-%d", expert);
            probe(gate.type, 2048, gate.data + expert * gate_bytes, up.data + expert * gate_bytes, fixture);
            probe(down.type, 512, down.data + expert * down_bytes, nullptr, fixture);
        }
    }
    sq::log("CPU Q2/Q3 MULTI-TOKEN SHARED UNPACK PASS: %llu cases, %llu exact float/guard comparisons",
            (unsigned long long)cases, (unsigned long long)checked);
    return 0;
}
