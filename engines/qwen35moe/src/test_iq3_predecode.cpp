// Standalone CPU-only lossless IQ3_S experiment; never initializes CUDA or writes model/cache files.
// Default: small exactness tests. Explicit --bench: bounded DRAM working set, six persistent workers.
// Usage: test_iq3_predecode [model.gguf] [--bench] [--mib 1024] [--rounds 3] [--variant MASK]
#include "common.hpp"
#include "cpu_moe.hpp"
#include "gguf.hpp"
#include "iq3_predecode_probe.hpp"
#include "native_iq/iq_avx2.hpp"
#include "quant.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace iq = strata::kernels::cpu;
namespace {
constexpr size_t MiB = 1024 * 1024;
constexpr int Rows = 35, Capacity = Rows + 3, BenchRows = 2048, BenchBlocks = 2, Workers = 6, Chunk = 32;
uint64_t cases = 0, comparisons = 0;
uint64_t packed_comparisons = 0;
volatile uint64_t eviction_sink = 0;

uint64_t available_ram() {
#ifdef _WIN32
    MEMORYSTATUSEX status{}; status.dwLength = sizeof(status);
    SQ_CHECK(GlobalMemoryStatusEx(&status), "cannot read available physical RAM");
    return status.ullAvailPhys;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    SQ_CHECK(pages > 0 && page_size > 0, "cannot read available physical RAM");
    return (uint64_t)pages * (uint64_t)page_size;
#endif
}

void initialize_activations(std::array<std::vector<sq::Q8K>, 4>& a, int nb, uint32_t seed) {
    std::mt19937 rng(seed);
    for (int t = 0; t < 4; ++t) {
        a[t].resize(nb);
        for (auto& b : a[t]) {
            std::memset(&b, 0, sizeof(b));
            b.d = (float)(1 + rng() % 7) / 4096;
            for (int k = 0; k < 256; ++k)
                b.nat[k] = t == 0 ? 0 : t == 1 ? (k & 1 ? -128 : 127) : (int8_t)((int)(rng() % 256) - 128);
        }
    }
}

void exact_probe(const uint8_t* source, int cols, const char* fixture) {
    const int nb = cols / 256;
    std::vector<iq3_probe::Block> decoded((size_t)Rows * nb);
    std::vector<iq3_probe::NibbleBlock> nibbles(decoded.size());
    iq3_probe::decode(source, decoded.data(), decoded.size());
    iq3_probe::decode_nibbles(source, nibbles.data(), nibbles.size());
    for (size_t i = 0; i < decoded.size(); ++i) {
        SQ_CHECK(decoded[i].d == nibbles[i].d &&
                 std::memcmp(decoded[i].subscales, nibbles[i].subscales, 8) == 0, "nibble metadata changed");
        for (int h = 0; h < 8; ++h) for (int k = 0; k < 32; ++k) {
            const int code = (nibbles[i].coefficients[h * 16 + (k & 15)] >> (k < 16 ? 0 : 4)) & 15;
            const int coefficient = (2 * (code & 7) + 1) * (code & 8 ? -1 : 1);
            SQ_CHECK(decoded[i].coefficients[h * 32 + k] == coefficient, "nibble coefficient changed");
            ++packed_comparisons;
        }
    }
    std::array<std::vector<sq::Q8K>, 4> a;
    initialize_activations(a, nb, 0x105061u + cols);
    const sq::DotFn legacy_dot = sq::dot_fn(sq::T_IQ3_S);
    const int maps[][4] = {{0, 1, 2, 3}, {1, 3, 0, 2}, {2, 2, 3, 2}, {3, 1, 2, 0}};
    for (int nt = 1; nt <= 4; ++nt) for (int variant : {0, 1, 4, 5, 8, 9, 12, 13})
        for (const auto& map : maps) for (int partial = 0; partial < 2; ++partial) {
            const int r0 = partial ? 3 : 0, r1 = partial ? Rows - 2 : Rows;
            const void* native_a[4]; const sq::Q8K* decoded_a[4];
            float *old_out[4], *new_out[4], *nibble_out[4];
            std::array<std::array<float, Capacity>, 4> old_values, new_values, nibble_values;
            for (int t = 0; t < 4; ++t) {
                native_a[t] = decoded_a[t] = a[map[t]].data();
                old_values[t].fill(123456.25f); new_values[t].fill(123456.25f); nibble_values[t].fill(123456.25f);
                old_out[t] = old_values[t].data(); new_out[t] = new_values[t].data(); nibble_out[t] = nibble_values[t].data();
            }
            iq::iq256_rows_v(variant, sq::T_IQ3_S, source, (size_t)nb * 110, cols,
                             native_a, nt, old_out, r0, r1);
            iq3_probe::rows(decoded.data(), nb, decoded_a, nt, new_out, r0, r1);
            iq3_probe::rows_nibbles(nibbles.data(), nb, decoded_a, nt, nibble_out, r0, r1);
            for (int t = 0; t < 4; ++t) for (int r = 0; r < Capacity; ++r) {
                SQ_CHECK(std::isfinite(new_out[t][r]) &&
                         std::memcmp(&new_out[t][r], &old_out[t][r], sizeof(float)) == 0,
                         "%s cols%d NT%d variant%d token%d row%d partial%d differs: %.9g / %.9g",
                         fixture, cols, nt, variant, t, r, partial, old_out[t][r], new_out[t][r]);
                ++comparisons;
                if (nt == 1 && t == 0 && r >= r0 && r < r1) {
                    const float legacy = legacy_dot(source + (size_t)r * nb * 110, decoded_a[0], nb);
                    SQ_CHECK(std::memcmp(&legacy, &nibble_out[0][r], sizeof(float)) == 0,
                             "%s legacy NT1 function-pointer dot differs at row%d variant%d", fixture, r, variant);
                    ++comparisons;
                }
                SQ_CHECK(std::isfinite(nibble_out[t][r]) &&
                         std::memcmp(&nibble_out[t][r], &old_out[t][r], sizeof(float)) == 0,
                         "%s nibble cols%d NT%d variant%d token%d row%d partial%d differs: %.9g / %.9g",
                         fixture, cols, nt, variant, t, r, partial, old_out[t][r], nibble_out[t][r]);
                ++comparisons;
            }
            ++cases;
        }
    sq::log("IQ3 predecode exact: %s cols%d native/int8/nibble NT1..4 scalar/gather/inline/compact signed-edge/gapped/duplicate/guards PASS",
            fixture, cols);
}

struct SourceExpert { const uint8_t* bytes; int layer; int expert; };
std::vector<SourceExpert> source_experts(const sq::GgufFile& gguf) {
    std::vector<SourceExpert> result;
    for (int layer = 0; layer < 40; ++layer) {
        const auto* d = gguf.tensor("blk." + std::to_string(layer) + ".ffn_down_exps.weight");
        if (!d || d->type != sq::T_IQ3_S) continue;
        SQ_CHECK(d->ne.size() == 3 && d->ne[0] == 512 && d->ne[1] == BenchRows && d->ne[2] == 256,
                 "unexpected IQ3 down shape");
        for (int expert = 0; expert < 256; ++expert)
            result.push_back({d->data + (size_t)expert * BenchRows * BenchBlocks * 110, layer, expert});
    }
    return result;
}

// Persistent workers exclude thread creation from the measured phases. Barrier publication gives each
// phase an immutable function; the host sleeps at completion instead of competing with its six workers.
class Pool {
    std::barrier<> start_{Workers + 1}, done_{Workers + 1};
    std::vector<std::thread> threads_;
    std::function<void()> work_;
    bool stop_ = false;
public:
    Pool() {
        for (int t = 0; t < Workers; ++t) threads_.emplace_back([this] {
            for (;;) {
                start_.arrive_and_wait();
                if (stop_) return;
                work_();
                done_.arrive_and_wait();
            }
        });
    }
    ~Pool() {
        stop_ = true;
        start_.arrive_and_wait();
        for (auto& t : threads_) t.join();
    }
    double run(std::function<void()> fn) {
        work_ = std::move(fn);
        const double begin = sq::now_ms();
        start_.arrive_and_wait();
        done_.arrive_and_wait();
        return sq::now_ms() - begin;
    }
};

uint64_t checksum(const std::vector<float>& a) {
    uint64_t hash = 1469598103934665603ull;
    for (float v : a) {
        uint32_t bits; std::memcpy(&bits, &v, sizeof(bits));
        hash = (hash ^ bits) * 1099511628211ull;
    }
    return hash;
}

void benchmark(sq::GgufFile& gguf, size_t limit_mib, int rounds, int variant) {
    auto sources = source_experts(gguf);
    const size_t blocks_per_expert = BenchRows * BenchBlocks;
    const size_t decoded_expert_bytes = blocks_per_expert * sizeof(iq3_probe::Block);
    const size_t native_expert_bytes = blocks_per_expert * 110;
    const size_t nibble_expert_bytes = blocks_per_expert * sizeof(iq3_probe::NibbleBlock);
    const size_t experts = std::min(sources.size(), limit_mib * MiB / decoded_expert_bytes);
    SQ_CHECK(experts >= 200, "benchmark needs at least 200 real experts to exceed LLC substantially");
    const size_t native_bytes = experts * native_expert_bytes, decoded_bytes = experts * decoded_expert_bytes;
    const size_t nibble_bytes = experts * nibble_expert_bytes;
    const size_t output_values = experts * 4 * BenchRows, output_bytes = output_values * sizeof(float);
    const size_t activation_blocks = experts * 4 * BenchBlocks;
    const size_t activation_bytes = activation_blocks * sizeof(sq::Q8K);
    const size_t eviction_bytes = 64 * MiB;
    // Include a second native-size allowance for original GGUF pages touched during copying, plus4GiB.
    const uint64_t guard = decoded_bytes + nibble_bytes + 2 * native_bytes + 3 * output_bytes + activation_bytes + eviction_bytes + 4096ull * MiB;
    const uint64_t available_before = available_ram();
    SQ_CHECK(available_before >= guard, "RAM guard: need %.1fMiB including4GiB headroom, available%.1fMiB",
             (double)guard / MiB, (double)available_before / MiB);
    sq::log("IQ3 PREDECODE BENCH CONFIG experts=%llu rows=%d blocks/row=%d workers=%d affinity=OS-default native_variant=%d",
            (unsigned long long)experts, BenchRows, BenchBlocks, Workers, variant);
    sq::log("IQ3 PREDECODE BENCH MEMORY native_bytes=%llu int8_bytes=%llu nibble_bytes=%llu all_output_bytes=%llu activation_bytes=%llu eviction_bytes=%llu available_before=%llu",
            (unsigned long long)native_bytes, (unsigned long long)decoded_bytes, (unsigned long long)nibble_bytes,
            (unsigned long long)(3 * output_bytes), (unsigned long long)activation_bytes, (unsigned long long)eviction_bytes,
            (unsigned long long)available_before);
    std::mt19937 rng(610231);
    std::shuffle(sources.begin(), sources.end(), rng);
    std::vector<uint8_t> native(native_bytes);
    std::vector<iq3_probe::Block> decoded(experts * blocks_per_expert);
    std::vector<iq3_probe::NibbleBlock> nibbles(experts * blocks_per_expert);
    std::array<std::vector<float>, 3> output{
        std::vector<float>(output_values), std::vector<float>(output_values), std::vector<float>(output_values)};
    std::vector<sq::Q8K> activations(activation_blocks);
    std::vector<uint8_t> eviction(eviction_bytes, 1);
    const double copy_start = sq::now_ms();
    for (size_t e = 0; e < experts; ++e)
        std::memcpy(native.data() + e * native_expert_bytes, sources[e].bytes, native_expert_bytes);
    const double copy_ms = sq::now_ms() - copy_start;
    // No model pointers survive this point. The timed controls use identical resident copies, not disk I/O.
    sources.clear(); gguf.close();
    const double build_start = sq::now_ms();
    iq3_probe::decode(native.data(), decoded.data(), decoded.size());
    const double build_ms = sq::now_ms() - build_start;
    const double nibble_start = sq::now_ms();
    iq3_probe::decode_nibbles(native.data(), nibbles.data(), nibbles.size());
    const double nibble_build_ms = sq::now_ms() - nibble_start;
    sq::log("IQ3 PREDECODE BUILD native_copy_ms=%.3f int8_build_ms=%.3f nibble_build_ms=%.3f available_after=%llu",
            copy_ms, build_ms, nibble_build_ms, (unsigned long long)available_ram());
    sq::log("BENCH NOTE: identical experts/rows in all three arms; --mib sizes the int8 arm only. Each timed pass visits every expert once in a newly shuffled order;64MiB cache sweep precedes each arm. Latin-balanced arm positions; resident DRAM test, not cold-disk or model throughput. Byte rates are logical weight bytes, not measured memory-controller traffic; activation/output traffic is additional.");
    // Down activations differ by expert. Both arms share the same distinct quantized vectors; do not make
    // every expert reuse four artificially hot vectors. Preparation is outside all measured intervals.
    for (auto& block : activations) {
        std::memset(&block, 0, sizeof(block));
        block.d = (float)(1 + rng() % 7) / 4096;
        for (int k = 0; k < 256; ++k) block.nat[k] = (int8_t)((int)(rng() % 255) - 127);
    }
    std::vector<size_t> order(experts); std::iota(order.begin(), order.end(), 0);
    std::atomic<size_t> next{0};
    constexpr size_t chunks_per_expert = BenchRows / Chunk;
    Pool pool;
    const char* arm_names[] = {"native", "int8", "nibble"};
    const size_t arm_bytes[] = {native_bytes, decoded_bytes, nibble_bytes};
    const int arm_order[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {1, 0, 2}, {0, 2, 1}};
    for (int nt = 1; nt <= 4; ++nt) {
        std::array<std::vector<double>, 3> samples;
        for (int round = 0; round < rounds; ++round) {
            std::shuffle(order.begin(), order.end(), rng);
            for (int position = 0; position < 3; ++position) {
                const int arm = arm_order[round][position];
                uint64_t touched = 0;
                for (size_t i = 0; i < eviction.size(); i += 64) {
                    ++eviction[i]; touched += eviction[i];
                }
                eviction_sink = touched;
                next.store(0, std::memory_order_relaxed);
                const double ms = pool.run([&] {
                    for (;;) {
                        const size_t task = next.fetch_add(1, std::memory_order_relaxed);
                        if (task >= experts * chunks_per_expert) break;
                        const size_t e = order[task / chunks_per_expert];
                        const int r0 = (int)(task % chunks_per_expert) * Chunk;
                        const void* native_a[4]; const sq::Q8K* decoded_a[4];
                        float* out[4];
                        for (int t = 0; t < 4; ++t) {
                            out[t] = output[arm].data() + (e * 4 + t) * BenchRows;
                            native_a[t] = decoded_a[t] = activations.data() + (e * 4 + t) * BenchBlocks;
                        }
                        if (arm == 2)
                            iq3_probe::rows_nibbles(nibbles.data() + e * blocks_per_expert, BenchBlocks,
                                                    decoded_a, nt, out, r0, r0 + Chunk);
                        else if (arm == 1)
                            iq3_probe::rows(decoded.data() + e * blocks_per_expert, BenchBlocks,
                                            decoded_a, nt, out, r0, r0 + Chunk);
                        else
                            iq::iq256_rows_v(variant, sq::T_IQ3_S, native.data() + e * native_expert_bytes,
                                             BenchBlocks * 110, BenchBlocks * 256, native_a, nt, out, r0, r0 + Chunk);
                    }
                });
                samples[arm].push_back(ms);
                sq::log("IQ3 PREDECODE SAMPLE NT%d round%d arm=%s ms=%.3f logical_weight_GBps=%.3f",
                        nt, round, arm_names[arm], ms, (double)arm_bytes[arm] / (ms * 1.e6));
            }
            SQ_CHECK(std::memcmp(output[0].data(), output[1].data(), output_bytes) == 0,
                     "DRAM benchmark output differs at NT%d round%d", nt, round);
            SQ_CHECK(std::memcmp(output[0].data(), output[2].data(), output_bytes) == 0,
                     "DRAM nibble benchmark output differs at NT%d round%d", nt, round);
            const uint64_t old_hash = checksum(output[0]), new_hash = checksum(output[1]), nibble_hash = checksum(output[2]);
            SQ_CHECK(old_hash == new_hash && old_hash == nibble_hash, "DRAM checksum differs");
            sq::log("IQ3 PREDECODE CHECK NT%d round%d exact_bytes=%llu checksum=%016llx",
                    nt, round, (unsigned long long)output_bytes, (unsigned long long)new_hash);
        }
        for (auto& s : samples) std::sort(s.begin(), s.end());
        const auto median = [](const std::vector<double>& s) {
            return (s[(s.size() - 1) / 2] + s[s.size() / 2]) * .5;
        };
        const double native_ms = median(samples[0]), decoded_ms = median(samples[1]), nibble_ms = median(samples[2]);
        sq::log("IQ3 PREDECODE SUMMARY NT%d rounds=%d native_ms=%.3f int8_ms=%.3f nibble_ms=%.3f int8_speedup=%.4f nibble_speedup=%.4f int8_build_ms=%.3f nibble_build_ms=%.3f",
                nt, rounds, native_ms, decoded_ms, nibble_ms, native_ms / decoded_ms, native_ms / nibble_ms,
                build_ms, nibble_build_ms);
    }
}
}

int main(int argc, char** argv) {
    SQ_CHECK(sq::cpu_has_avx2(), "AVX2/F16C/FMA required");
    const char* model = nullptr;
    bool bench = false;
    size_t limit_mib = 1024;
    int rounds = 3, variant = iq::iq256_variant();
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bench") bench = true;
        else if (arg == "--mib" || arg == "--rounds" || arg == "--variant") {
            SQ_CHECK(i + 1 < argc, "%s needs an integer", arg.c_str());
            char* end = nullptr; const long v = std::strtol(argv[++i], &end, 10);
            SQ_CHECK(end != argv[i] && end && *end == '\0', "invalid integer");
            if (arg == "--mib") { SQ_CHECK(v >= 256 && v <= 1024, "--mib range256..1024"); limit_mib = (size_t)v; }
            else if (arg == "--rounds") { SQ_CHECK(v == 3 || v == 6, "--rounds must be3 or6 for balanced three-arm order"); rounds = (int)v; }
            else { SQ_CHECK(v >= 0 && (v & ~iq::iq256_variants()) == 0, "unsupported variant"); variant = (int)v; }
        } else { SQ_CHECK(!model && arg.rfind("--", 0) != 0, "unknown argument%s", arg.c_str()); model = argv[i]; }
    }
    SQ_CHECK(!bench || model, "--bench requires real model GGUF");
    std::mt19937 rng(610231);
    for (int cols : {256, 512, 2048}) {
        std::vector<uint8_t> source((size_t)Rows * (cols / 256) * 110);
        for (auto& b : source) b = (uint8_t)rng();
        for (size_t b = 0; b < source.size(); b += 110) {
            const uint16_t d = sq::f32_to_fp16((float)((int)(rng() % 9) - 4) / 4096);
            std::memcpy(source.data() + b, &d, 2);
            for (int h = 0; h < 4; ++h) source[b + 106 + h] = (uint8_t)((b / 110) * 4 + h);
        }
        exact_probe(source.data(), cols, "random/all-subscales");
    }
    {
        // Exercise every source grid entry with both sign polarities, plus every packed subscale byte.
        // This guards the exact sixteen-value alphabet assumed by the lossless nibble representation.
        std::vector<uint8_t> source((size_t)Rows * 2 * 110);
        const uint16_t d = sq::f32_to_fp16(1.f / 4096);
        for (size_t i = 0; i < Rows * 2; ++i) {
            uint8_t* b = source.data() + i * 110;
            std::memcpy(b, &d, 2);
            for (int h = 0; h < 8; ++h) {
                for (int g = 0; g < 8; ++g) {
                    const int index = (int)((i * 64 + h * 8 + g) % 512);
                    b[2 + h * 8 + g] = (uint8_t)index;
                    b[66 + h] |= (uint8_t)((index >> 8) << g);
                }
                for (int k = 0; k < 4; ++k) b[74 + 4 * h + k] = (i / 8) & 1 ? 0xAA : 0x55;
            }
            for (int s = 0; s < 4; ++s) b[106 + s] = (uint8_t)(i * 4 + s);
        }
        exact_probe(source.data(), 512, "all512-grids/both-signs/all-subscales");
    }
    sq::GgufFile gguf;
    if (model) {
        std::string error; SQ_CHECK(gguf.open(model, error), "%s", error.c_str());
        const auto experts = source_experts(gguf);
        SQ_CHECK(experts.size() >= 256, "no real IQ3_S down experts");
        for (size_t i : {size_t(0), experts.size() / 2 + 17, experts.size() - 1}) {
            char label[80]; std::snprintf(label, sizeof(label), "real-layer%d-expert%d", experts[i].layer, experts[i].expert);
            exact_probe(experts[i].bytes, 512, label);
        }
    }
    sq::log("IQ3 PREDECODE EXACT PASS: %llu cases %llu float/guard comparisons %llu nibble/int8 coefficient comparisons",
            (unsigned long long)cases, (unsigned long long)comparisons, (unsigned long long)packed_comparisons);
    if (bench) benchmark(gguf, limit_mib, rounds, variant);
    return 0;
}
