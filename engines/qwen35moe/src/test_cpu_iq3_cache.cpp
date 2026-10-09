// Real CpuMoe off/on exactness and cache lifecycle. CPU-only; optional GGUF is mapped read-only.
#include "common.hpp"
#include "cpu_iq3_cache.hpp"
#include "cpu_moe.hpp"
#include "gguf.hpp"
#include "quant.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int NE = 3, E = 2048, FF = 512;
using Output = std::array<std::array<float, E>, sq::kMaxT>;
struct Fixture {
    std::vector<uint8_t> bytes;
    sq::ExpertLayerDesc desc;
    const char* name;
};
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
struct SavedEnv {
    const char* name; bool present; std::string value;
    explicit SavedEnv(const char* n) : name(n), present(std::getenv(n) != nullptr), value(present ? std::getenv(n) : "") {}
    ~SavedEnv() { set_env(name, present ? value.c_str() : nullptr); }
};
Fixture synthetic() {
    Fixture f;
    f.name = "random";
    const size_t gu = (size_t)sq::row_bytes(sq::T_IQ2_S, E) * FF;
    const size_t down = (size_t)sq::row_bytes(sq::T_IQ3_S, FF) * E;
    f.bytes.resize(NE * (2 * gu + down));
    std::mt19937 rng(105061);
    for (auto& b : f.bytes) b = (uint8_t)rng();
    for (int e = 0; e < NE; ++e) for (int matrix = 0; matrix < 3; ++matrix) {
        const size_t offset = (size_t)e * (2 * gu + down) + matrix * gu;
        const size_t block = matrix == 2 ? 110 : 82, length = matrix == 2 ? down : gu;
        for (size_t i = 0; i < length; i += block) {
            const uint16_t d = sq::f32_to_fp16((float)(1 + rng() % 4) / 4096);
            std::memcpy(f.bytes.data() + offset + i, &d, 2);
        }
    }
    f.desc = {f.bytes.data(), (int64_t)(2 * gu + down), (int64_t)gu, sq::T_IQ2_S, sq::T_IQ3_S, NE};
    return f;
}
Fixture real(const char* model) {
    sq::GgufFile gguf; std::string error;
    SQ_CHECK(gguf.open(model, error), "%s", error.c_str());
    for (int layer = 0; layer < 40; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto& down = gguf.need(prefix + "ffn_down_exps.weight");
        if (down.type != sq::T_IQ3_S) continue;
        const auto& gate = gguf.need(prefix + "ffn_gate_exps.weight");
        const auto& up = gguf.need(prefix + "ffn_up_exps.weight");
        SQ_CHECK(gate.type == up.type && gate.ne[0] == E && gate.ne[1] == FF && gate.ne[2] == 256 &&
                 down.ne[0] == FF && down.ne[1] == E && down.ne[2] == 256, "unexpected real expert shape");
        const size_t gu = (size_t)sq::row_bytes(gate.type, E) * FF;
        const size_t d = (size_t)sq::row_bytes(down.type, FF) * E;
        Fixture f; f.name = "real-experts-0-17-255";
        f.bytes.resize(NE * (2 * gu + d));
        const int source_ids[NE] = {0, 17, 255};
        for (int i = 0; i < NE; ++i) {
            uint8_t* target = f.bytes.data() + (size_t)i * (2 * gu + d);
            std::memcpy(target, gate.data + source_ids[i] * gu, gu);
            std::memcpy(target + gu, up.data + source_ids[i] * gu, gu);
            std::memcpy(target + 2 * gu, down.data + source_ids[i] * d, d);
        }
        f.desc = {f.bytes.data(), (int64_t)(2 * gu + d), (int64_t)gu, gate.type, down.type, NE};
        return f;
    }
    sq::die("model has no IQ3_S down expert");
}

void run_fixture(Fixture& f, int workers, uint64_t& checked) {
    auto mb = std::make_unique<sq::Mailbox[]>(2);
    auto result = std::make_unique<sq::Result[]>(2);
    auto cpu = std::make_unique<sq::CpuMoe>();
    std::vector<Output> expected;
    const std::vector<sq::ExpertLayerDesc> layers{f.desc, sq::ExpertLayerDesc{}};
    const size_t cache_bytes = (size_t)NE * sq::CpuIq3DownCache::BlocksPerExpert * sizeof(sq::native_iq::Iq3NibbleBlock);
    for (int lifecycle = 0; lifecycle < 4; ++lifecycle) {
        const bool enabled = (lifecycle & 1) != 0;
        set_env("STRATA_CPU_IQ3_NIBBLE", enabled ? "1" : "0");
        cpu->start(workers, 2, layers, mb.get(), result.get(), false);
        SQ_CHECK(cpu->iq3_down_cache_bytes() == (enabled ? cache_bytes : 0), "cache flag/byte accounting differs");
        size_t case_id = 0;
        for (int nt = 1; nt <= 4; ++nt) for (int pattern = 0; pattern < 5; ++pattern) {
            std::memset(&mb[0], 0, sizeof(sq::Mailbox));
            std::memset(&mb[1], 0, sizeof(sq::Mailbox));
            mb[0].n_tok = nt; mb[0].n_miss = pattern == 4 ? 0 : NE;
            mb[1].n_tok = nt; mb[1].n_miss = 0;
            const int ids[NE] = {2, 0, 1}; // preserve caller order, including last allocated expert
            for (int j = 0; j < NE; ++j) {
                mb[0].ids[j] = ids[j];
                for (int t = 0; t < nt; ++t) {
                    const bool active = pattern == 0 || (pattern == 1 ? j == t % NE :
                                        pattern == 2 ? ((j + t) % 2 == 0) : j != 1);
                    mb[0].w[j][t] = active ? (float)(j + 1) * (t & 1 ? -.125f : .125f) : 0.f;
                }
            }
            for (int t = 0; t < nt; ++t) for (int k = 0; k < E; ++k)
                mb[0].x[t][k] = pattern == 3 ? 0.f : (float)(((k * 37 + t * 53 + pattern) % 255) - 127) / 128;
            for (int layer = 0; layer < 2; ++layer)
                for (auto& row : result[layer].out) for (float& v : row) v = 123456.25f;
            if (pattern & 1) {
                mb[0].ready_seq = mb[1].ready_seq = (uint32_t)(1 + case_id + lifecycle * 100);
                cpu->begin_step(mb[0].ready_seq, 0, 2);
                SQ_CHECK(cpu->end_step(), "published CPU test step timed out");
            } else {
                cpu->begin_layer(0); cpu->end_layer();
                cpu->begin_layer(1); cpu->end_layer();
            }
            Output output;
            std::memcpy(output.data(), result[0].out, sizeof(output));
            if (lifecycle == 0) expected.push_back(output);
            else SQ_CHECK(std::memcmp(output.data(), expected[case_id].data(), sizeof(output)) == 0,
                          "%s workers%d life%d NT%d pattern%d off/on output differs", f.name, workers, lifecycle, nt, pattern);
            for (int t = 0; t < sq::kMaxT; ++t) for (int k = 0; k < E; ++k) {
                SQ_CHECK(std::isfinite(output[t][k]), "nonfinite CPU result");
                if (t >= nt || pattern == 4) SQ_CHECK(output[t][k] == 123456.25f, "inactive output modified");
                SQ_CHECK(result[1].out[t][k] == 123456.25f, "empty/fallback layer modified");
                ++checked;
            }
            if (nt == 1 && pattern < 4) {
                std::array<float, E> sync_output;
                float w[NE]; for (int j = 0; j < NE; ++j) w[j] = mb[0].w[j][0];
                cpu->compute_layer_sync(0, mb[0].x[0], mb[0].ids, w, NE, sync_output.data());
                SQ_CHECK(std::memcmp(sync_output.data(), output[0].data(), sizeof(sync_output)) == 0,
                         "legacy sync T1 result differs");
                checked += E;
            }
            ++case_id;
        }
        SQ_CHECK(enabled ? cpu->iq3_down_cache_rows() > 0 : cpu->iq3_down_cache_rows() == 0,
                 "cache coverage counter did not match enabled state");
        cpu->stop();
        SQ_CHECK(cpu->iq3_down_cache_bytes() == 0, "cache retained after worker join");
        cpu->stop(); // repeated idle shutdown is safe
    }
    sq::log("CPU IQ3 cache %s workers%d NT1..4 shuffled/gapped/zero/guards/restart/sync exact PASS", f.name, workers);
}
}

int main(int argc, char** argv) {
    SQ_CHECK(sq::cpu_force_isa("avx2"), "AVX2 required");
    SavedEnv cache("STRATA_CPU_IQ3_NIBBLE"), idle("STRATA_CPU_WORKER_SPINS"), completion("STRATA_CPU_COMPLETION_SPINS");
    set_env("STRATA_CPU_WORKER_SPINS", "0"); set_env("STRATA_CPU_COMPLETION_SPINS", "0");
    uint64_t checked = 0;
    auto random = synthetic();
    size_t bytes = 0; std::string error;
    auto invalid = random.desc; invalid.n_expert = 0;
    SQ_CHECK(!sq::CpuIq3DownCache::estimate_bytes({invalid}, bytes, error), "missing extent accepted");
    SQ_CHECK(sq::CpuIq3DownCache::estimate_bytes({random.desc, sq::ExpertLayerDesc{}}, bytes, error) &&
             bytes == NE * sq::CpuIq3DownCache::BlocksPerExpert * sizeof(sq::native_iq::Iq3NibbleBlock), "estimate mismatch");
    for (int workers : {1, 6}) run_fixture(random, workers, checked);
    if (argc > 1) { auto model = real(argv[1]); for (int workers : {1, 6}) run_fixture(model, workers, checked); }
    sq::log("CPU IQ3 CACHE EXACT PASS: %llu float/guard checks", (unsigned long long)checked);
    return 0;
}
