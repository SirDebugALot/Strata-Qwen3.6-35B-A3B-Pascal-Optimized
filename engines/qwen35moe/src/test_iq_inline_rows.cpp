// CPU-only exact old-versus-inlined/compact-scale/paired-GU IQ chunks. Never initializes CUDA or writes model/cache files.
// Usage: test_iq_inline_rows [model.gguf] [--bench]; benchmark is optional and single-threaded.
#include "common.hpp"
#include "cpu_moe.hpp"
#include "gguf.hpp"
#include "native_iq/iq_avx2.hpp"
#include "quant.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>

namespace iq = strata::kernels::cpu;
namespace {
constexpr int Rows = 35, Capacity = Rows + 3;
uint64_t cases = 0, checked = 0;

std::vector<uint8_t> random_weights(uint32_t type, int cols, std::mt19937& rng) {
    const size_t block_bytes = (size_t)sq::row_bytes(type, 256);
    std::vector<uint8_t> result((size_t)Rows * sq::row_bytes(type, cols));
    for (uint8_t& b : result) b = (uint8_t)rng();
    for (size_t b = 0; b < result.size(); b += block_bytes) {
        const uint16_t d = sq::f32_to_fp16((float)((int)(rng() % 9) - 4) / 4096);
        std::memcpy(result.data() + b, &d, 2);
        // Every packed IQ2 scale byte and IQ3 nibble combination occurs even at the smaller row width.
        if (type == sq::T_IQ2_S)
            for (int h = 0; h < 8; ++h) result[b + 74 + h] = (uint8_t)((b / block_bytes) * 8 + h);
        if (type == sq::T_IQ3_S)
            for (int h = 0; h < 4; ++h) result[b + 106 + h] = (uint8_t)((b / block_bytes) * 4 + h);
    }
    return result;
}

void probe(uint32_t type, int cols, const uint8_t* weights, const uint8_t* up, const char* fixture, bool bench) {
    const size_t rb = (size_t)sq::row_bytes(type, cols), up_offset = Rows * rb + 18;
    std::vector<uint8_t> blob(up_offset + Rows * rb);
    std::memcpy(blob.data(), weights, Rows * rb);
    if (up) std::memcpy(blob.data() + up_offset, up, Rows * rb);
    std::array<std::vector<sq::Q8K>, 4> activations;
    std::mt19937 rng(610231 + type + cols);
    for (int t = 0; t < 4; ++t) {
        activations[t].resize(cols / 256);
        for (sq::Q8K& block : activations[t]) {
            std::memset(&block, 0, sizeof(block));
            block.d = (float)(1 + rng() % 7) / 4096;
            for (int i = 0; i < 256; ++i)
                block.nat[i] = t == 0 ? 0 : t == 2 ? (i & 1 ? -128 : 127) : (int8_t)((int)(rng() % 256) - 128);
        }
    }
    const int mappings[][4] = {{0, 1, 2, 3}, {3, 1, 0, 2}, {1, 1, 3, 1}, {2, 0, 3, 1}};
    struct VariantPair { int old_variant, new_variant; };
    const VariantPair variants[] = {
        {0, iq::kIq256InlineRows}, {iq::kIq256Gather, iq::kIq256Gather | iq::kIq256InlineRows},
        {0, iq::kIq256CompactScales}, {iq::kIq256Gather, iq::kIq256Gather | iq::kIq256CompactScales},
        {iq::kIq256InlineRows, iq::kIq256InlineRows | iq::kIq256CompactScales},
        {iq::kIq256Gather | iq::kIq256InlineRows, iq::kIq256Gather | iq::kIq256InlineRows | iq::kIq256CompactScales},
        // Pairing is independent of gather/inline/compact. At NT2..4 and in the dot path it must be ignored.
        {0, iq::kIq256PairGu},
        {iq::kIq256Gather, iq::kIq256Gather | iq::kIq256PairGu},
        {iq::kIq256InlineRows, iq::kIq256InlineRows | iq::kIq256PairGu},
        {iq::kIq256Gather | iq::kIq256InlineRows,
         iq::kIq256Gather | iq::kIq256InlineRows | iq::kIq256PairGu},
        {iq::kIq256CompactScales, iq::kIq256CompactScales | iq::kIq256PairGu},
        {iq::kIq256Gather | iq::kIq256CompactScales,
         iq::kIq256Gather | iq::kIq256CompactScales | iq::kIq256PairGu},
        {iq::kIq256InlineRows | iq::kIq256CompactScales,
         iq::kIq256InlineRows | iq::kIq256CompactScales | iq::kIq256PairGu},
        {iq::kIq256Gather | iq::kIq256InlineRows | iq::kIq256CompactScales,
         iq::kIq256Gather | iq::kIq256InlineRows | iq::kIq256CompactScales | iq::kIq256PairGu}
    };
    for (const auto& pair : variants) for (int nt = 1; nt <= 4; ++nt)
        for (const auto& mapping : mappings) for (int range = 0; range < 2; ++range) {
            const int variant = pair.new_variant;
            const int r0 = range ? 3 : 0, r1 = range ? Rows - 2 : Rows;
            const void* acts[4];
            float *outputs[4], *ref_outputs[4], *gu_outputs[4], *ref_gu_outputs[4];
            std::array<std::array<float, Capacity>, 4> actual, reference, actual_gu, reference_gu;
            for (int t = 0; t < 4; ++t) {
                acts[t] = activations[mapping[t]].data();
                outputs[t] = actual[t].data(); ref_outputs[t] = reference[t].data();
                gu_outputs[t] = actual_gu[t].data(); ref_gu_outputs[t] = reference_gu[t].data();
                actual[t].fill(123456.25f); reference[t].fill(123456.25f);
                actual_gu[t].fill(123456.25f); reference_gu[t].fill(123456.25f);
            }
            iq::iq256_rows_v(pair.old_variant, type, blob.data(), rb, cols, acts, nt, ref_outputs, r0, r1);
            iq::iq256_rows_v(pair.new_variant, type, blob.data(), rb, cols, acts, nt, outputs, r0, r1);
            if (up) {
                iq::iq256_gu_rows_v(pair.old_variant, type, blob.data(), rb, up_offset, cols, acts, nt, ref_gu_outputs, r0, r1);
                iq::iq256_gu_rows_v(pair.new_variant, type, blob.data(), rb, up_offset, cols,
                                    acts, nt, gu_outputs, r0, r1);
            }
            for (int t = 0; t < 4; ++t) for (int row = 0; row < Capacity; ++row) {
                SQ_CHECK(std::isfinite(actual[t][row]) &&
                         std::memcmp(&reference[t][row], &actual[t][row], sizeof(float)) == 0,
                         "%s type%u cols%d NT%d variant%d token%d row%d range%d dot differs: %.9g/%.9g",
                         fixture, type, cols, nt, variant, t, row, range, reference[t][row], actual[t][row]);
                ++checked;
                if (nt == 1 && t == 0 && row >= r0 && row < r1) {
                    // Singleton experts in a T>1 worker step previously called the plain dot wrapper even
                    // when gather was enabled. Require exact compatibility with that path as well.
                    const float single = sq::dot_row(type, blob.data() + row * rb, (const sq::Q8K*)acts[0], cols / 256);
                    SQ_CHECK(std::memcmp(&single, &actual[0][row], sizeof(float)) == 0,
                             "%s NT1 legacy wrapper differs, type%u variant%d row%d", fixture, type, variant, row);
                    ++checked;
                }
                if (up) {
                    SQ_CHECK(std::isfinite(actual_gu[t][row]) &&
                             std::memcmp(&reference_gu[t][row], &actual_gu[t][row], sizeof(float)) == 0,
                             "%s type%u cols%d NT%d variant%d token%d row%d range%d GU differs: %.9g/%.9g",
                             fixture, type, cols, nt, variant, t, row, range, reference_gu[t][row], actual_gu[t][row]);
                    ++checked;
                }
            }
            ++cases;
        }
    sq::log("IQ inline/compact/paired chunks: %s type%u cols%d T1..4/scalar-gather/inline-compact-pair/gapped/duplicate/range/guards%s exact=PASS",
            fixture, type, cols, up ? "/gate-up" : "/down");
    if (!bench) return;
    const void* acts[4]; float* outputs[4];
    std::array<std::array<float, Capacity>, 4> output;
    for (int t = 0; t < 4; ++t) { acts[t] = activations[t].data(); outputs[t] = output[t].data(); }
    constexpr int repeats = 128;
    for (const auto& pair : variants) for (int nt = 1; nt <= 4; ++nt) {
        if ((pair.new_variant & iq::kIq256PairGu) && (nt != 1 || !up)) continue;
        double timings[2] = {};
        // Alternate order over four short rounds to reduce ordering/turbo bias. This is a hot-row microbench,
        // not a DRAM-streaming or six-worker engine measurement; full matched requests remain decisive.
        for (int round = 0; round < 4; ++round) for (int order = 0; order < 2; ++order) {
            const int arm = order ^ (round & 1), variant = arm ? pair.new_variant : pair.old_variant;
            auto run = [&] {
                if (up) iq::iq256_gu_rows_v(variant, type, blob.data(), rb, up_offset, cols, acts, nt, outputs, 0, 32);
                else iq::iq256_rows_v(variant, type, blob.data(), rb, cols, acts, nt, outputs, 0, 32);
            };
            run();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < repeats; ++i) run();
            timings[arm] += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        }
        sq::log("IQ chunk BENCH %s type%u cols%d NT%d variants%d/%d %s old_ns/row=%.2f new_ns/row=%.2f ratio=%.4f",
                fixture, type, cols, nt, pair.old_variant, pair.new_variant, up ? "GU" : "down", timings[0] / (4 * repeats * 32),
                timings[1] / (4 * repeats * 32), timings[1] / timings[0]);
    }
}
} // namespace

int main(int argc, char** argv) {
    SQ_CHECK(sq::cpu_has_avx2(), "test requires AVX2/F16C CPU");
    const char* model = nullptr;
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else model = argv[i];
    }
    std::mt19937 rng(610231);
    for (uint32_t type : {sq::T_IQ2_S, sq::T_IQ3_S, sq::T_IQ4_XS}) for (int cols : {512, 2048}) {
        const auto weights = random_weights(type, cols, rng), up = random_weights(type, cols, rng);
        probe(type, cols, weights.data(), up.data(), "random", false);
    }
    if (model) {
        sq::GgufFile gguf; std::string error;
        SQ_CHECK(gguf.open(model, error), "%s", error.c_str());
        std::set<std::pair<uint32_t, uint32_t>> seen;
        for (int layer = 0; layer < 40; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto& gate = gguf.need(prefix + "ffn_gate_exps.weight");
            const auto& up = gguf.need(prefix + "ffn_up_exps.weight");
            const auto& down = gguf.need(prefix + "ffn_down_exps.weight");
            if (!seen.emplace(gate.type, down.type).second) continue;
            SQ_CHECK(gate.type == up.type && gate.ne[0] == 2048 && gate.ne[1] == 512 && gate.ne[2] == 256 &&
                     down.ne[0] == 512 && down.ne[1] == 2048 && down.ne[2] == 256, "unexpected Qwen expert shape");
            const size_t gb = (size_t)sq::row_bytes(gate.type, 2048) * 512;
            const size_t db = (size_t)sq::row_bytes(down.type, 512) * 2048;
            for (int expert : {0, 17, 255}) {
                char fixture[64]; std::snprintf(fixture, sizeof(fixture), "real-layer-%d-expert-%d", layer, expert);
                probe(gate.type, 2048, gate.data + expert * gb, up.data + expert * gb, fixture, bench && expert == 17);
                probe(down.type, 512, down.data + expert * db, nullptr, fixture, bench && expert == 17);
                if (expert == 17) {
                    // The actual model uses IQ3_S for down, not gate/up. Pair two disjoint real row ranges
                    // as GU operands to exercise the paired IQ3 decoder on source weights as well.
                    const size_t rb = (size_t)sq::row_bytes(down.type, 512);
                    probe(down.type, 512, down.data + expert * db, down.data + expert * db + Rows * rb,
                          "real-down-rows-paired-as-GU", false);
                }
            }
        }
    }
    sq::log("CPU IQ INLINE/COMPACT/PAIRED CHUNKS PASS: %llu cases, %llu exact float/guard comparisons",
            (unsigned long long)cases, (unsigned long long)checked);
    return 0;
}
