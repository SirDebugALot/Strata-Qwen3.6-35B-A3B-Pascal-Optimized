// CPU-only diagnostics regression: exact outputs across profile/worker/window settings, real routed-work
// conservation, empty jobs, partial windows, and stop/restart. No timer magnitude is used as a speed claim.
#include "common.hpp"
#include "cpu_moe.hpp"
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
void env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
struct SavedEnv {
    const char* name; bool present; std::string value;
    explicit SavedEnv(const char* n) : name(n), present(std::getenv(n) != nullptr), value(present ? std::getenv(n) : "") {}
    ~SavedEnv() { env(name, present ? value.c_str() : nullptr); }
};
struct Input {
    sq::Mailbox mailbox = {};
    uint64_t histogram[5] = {};
    uint64_t pairs = 0;
};
Input input(int nt, int pattern) {
    Input in;
    auto& m = in.mailbox;
    m.n_tok = nt; m.n_miss = pattern == 2 ? 0 : NE;
    const int ids[NE] = {2, 0, 1};
    for (int j = 0; j < NE; ++j) {
        m.ids[j] = ids[j];
        // Dense expert, alternating/gapped tokens, and a deliberately unused expert. Negative weights
        // are legal for this CPU contract and expose accidental reassociation in weighted reduction.
        for (int t = 0; t < nt; ++t)
            m.w[j][t] = pattern == 0 || (j != 2 && (j == 0 || (t & 1))) ?
                        (t & 1 ? -.0625f : .125f) * (j + 1) : 0.f;
    }
    for (int t = 0; t < nt; ++t) for (int k = 0; k < E; ++k)
        m.x[t][k] = (float)(((k * 37 + t * 53 + pattern * 17) % 257) - 128) / 128;
    // Expected occupancy comes directly from the independently chosen input pattern.
    if (pattern == 0) { in.histogram[nt] = NE; in.pairs = NE * nt; }
    if (pattern == 1) { ++in.histogram[nt]; ++in.histogram[nt / 2]; ++in.histogram[0]; in.pairs = nt + nt / 2; }
    return in;
}

void check_profile(const sq::CpuMoePhaseProfile& p, const std::vector<Input>& inputs, int skip, int limit,
                   bool enabled, int workers) {
    if (!enabled) {
        SQ_CHECK(p.seen_jobs == 0 && p.sampled_jobs == 0 && p.groups.empty() && !p.complete,
                 "disabled diagnostics collected data");
        return;
    }
    const int end = std::min((int)inputs.size(), skip + limit);
    const int sampled = std::max(0, end - skip);
    SQ_CHECK(p.seen_jobs == inputs.size() && p.sampled_jobs == (uint64_t)sampled && p.complete == (sampled == limit),
             "sample window/complete boundary differs");
    uint64_t total_jobs = 0;
    for (const auto& group : p.groups) {
        uint64_t jobs = 0, missing = 0, pairs = 0, hist[5] = {}, nonempty = 0;
        for (int i = skip; i < end; ++i) if (inputs[i].mailbox.n_tok == group.n_tok) {
            ++jobs; missing += inputs[i].mailbox.n_miss; pairs += inputs[i].pairs;
            nonempty += inputs[i].mailbox.n_miss > 0;
            for (int nt = 0; nt <= 4; ++nt) hist[nt] += inputs[i].histogram[nt];
        }
        SQ_CHECK(group.t_gu == sq::T_IQ2_S && group.t_down == sq::T_IQ3_S && group.jobs == jobs &&
                 group.missing_experts == missing && group.routed_pairs == pairs &&
                 std::memcmp(group.expert_nt, hist, sizeof(hist)) == 0, "routed-work histogram differs");
        SQ_CHECK(group.workers.gu_row_pairs == pairs * FF && group.workers.down_row_pairs == pairs * E,
                 "profile did not account for each routed output row exactly once");
        SQ_CHECK(group.workers.gu_quant_tokens >= nonempty * group.n_tok &&
                 group.workers.gu_quant_tokens <= nonempty * group.n_tok * workers &&
                 group.workers.down_quant_pairs >= pairs && group.workers.down_quant_pairs <= pairs * workers,
                 "quantization coverage is inconsistent with worker/chunk ownership");
        uint64_t phases = 0;
        for (int phase = 0; phase < sq::CpuWorkerTotal; ++phase) phases += group.workers.ns[phase];
        SQ_CHECK(group.workers.ns[sq::CpuWorkerTotal] >= phases && group.dispatch_to_join_ns >= group.max_worker_ns,
                 "nested phase accounting exceeds enclosing interval");
        if (nonempty) {
            SQ_CHECK(group.workers.ns[sq::CpuPackGU] && group.workers.ns[sq::CpuGU] &&
                     group.workers.ns[sq::CpuPackDown] && group.max_worker_ns, "nonvacuous phases not measured");
            if (group.n_tok == 1)
                SQ_CHECK(group.workers.ns[sq::CpuDownFused] > 0 && group.workers.ns[sq::CpuDown] == 0,
                         "legacy fused down path misclassified");
            else
                SQ_CHECK(group.workers.ns[sq::CpuDown] > 0 && group.workers.ns[sq::CpuReduce] > 0 &&
                         group.workers.ns[sq::CpuDownFused] == 0, "grouped down/reduction phases missing");
        }
        total_jobs += group.jobs;
    }
    SQ_CHECK(total_jobs == p.sampled_jobs, "sample groups do not partition the window");
}
}

int main() {
    SQ_CHECK(sq::cpu_force_isa("avx2"), "AVX2 required");
    SavedEnv prof("STRATA_CPU_PHASE_PROFILE"), skip("STRATA_CPU_PHASE_PROFILE_SKIP"), jobs("STRATA_CPU_PHASE_PROFILE_JOBS"),
             cache("STRATA_CPU_IQ3_NIBBLE"), idle("STRATA_CPU_WORKER_SPINS"), wait("STRATA_CPU_COMPLETION_SPINS");
    env("STRATA_CPU_IQ3_NIBBLE", "0"); env("STRATA_CPU_WORKER_SPINS", "0"); env("STRATA_CPU_COMPLETION_SPINS", "0");
    const size_t gu = (size_t)sq::row_bytes(sq::T_IQ2_S, E) * FF;
    const size_t down = (size_t)sq::row_bytes(sq::T_IQ3_S, FF) * E;
    std::vector<uint8_t> bytes(NE * (2 * gu + down));
    std::mt19937 rng(2061105);
    for (auto& b : bytes) b = (uint8_t)rng();
    for (int e = 0; e < NE; ++e) for (int matrix = 0; matrix < 3; ++matrix) {
        const size_t block = matrix == 2 ? 110 : 82, length = matrix == 2 ? down : gu;
        const size_t offset = (size_t)e * (2 * gu + down) + matrix * gu;
        for (size_t i = 0; i < length; i += block) {
            const uint16_t d = sq::f32_to_fp16((float)(1 + rng() % 4) / 4096);
            std::memcpy(bytes.data() + offset + i, &d, 2);
        }
    }
    const std::vector<sq::ExpertLayerDesc> layers{{bytes.data(), (int64_t)(2 * gu + down), (int64_t)gu,
                                                 sq::T_IQ2_S, sq::T_IQ3_S, NE}};
    std::vector<Input> inputs;
    for (int nt = 1; nt <= 4; ++nt) for (int pattern = 0; pattern < 3; ++pattern) inputs.push_back(input(nt, pattern));
    auto mb = std::make_unique<sq::Mailbox[]>(1);
    auto result = std::make_unique<sq::Result[]>(1);
    auto cpu = std::make_unique<sq::CpuMoe>();
    std::vector<Output> expected;
    struct Run { int workers; bool enabled; int skip, limit; };
    const Run runs[] = {{1,false,0,12}, {1,true,0,12}, {6,true,2,7}, {6,true,2,100}, {1,true,100,7}, {6,false,0,12}};
    uint64_t checked = 0;
    for (const auto& run : runs) {
        env("STRATA_CPU_PHASE_PROFILE", run.enabled ? "1" : "0");
        env("STRATA_CPU_PHASE_PROFILE_SKIP", std::to_string(run.skip).c_str());
        env("STRATA_CPU_PHASE_PROFILE_JOBS", std::to_string(run.limit).c_str());
        cpu->start(run.workers, 1, layers, mb.get(), result.get(), false);
        SQ_CHECK(cpu->phase_profile().groups.empty(), "profile was retained across restart");
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::memcpy(&mb[0], &inputs[i].mailbox, sizeof(sq::Mailbox));
            for (auto& row : result[0].out) for (float& v : row) v = 123456.25f;
            cpu->begin_layer(0); cpu->end_layer();
            Output actual;
            std::memcpy(actual.data(), result[0].out, sizeof(actual));
            if (&run == &runs[0]) expected.push_back(actual);
            else SQ_CHECK(std::memcmp(actual.data(), expected[i].data(), sizeof(actual)) == 0,
                          "phase profile changed result/guard: workers%d enabled%d case%zu", run.workers, run.enabled, i);
            for (int t = 0; t < 4; ++t) for (int k = 0; k < E; ++k) {
                SQ_CHECK(std::isfinite(actual[t][k]), "nonfinite output");
                if (t >= mb[0].n_tok || mb[0].n_miss == 0) SQ_CHECK(actual[t][k] == 123456.25f, "inactive output overwritten");
                ++checked;
            }
        }
        check_profile(cpu->phase_profile(), inputs, run.skip, run.limit, run.enabled, run.workers);
        const auto before = cpu->phase_profile();
        std::vector<int> callbacks(run.workers, 0);
        cpu->parallel_for_workers([&](int tid) { callbacks[tid] = tid + 1; });
        for (int tid = 0; tid < run.workers; ++tid) SQ_CHECK(callbacks[tid] == tid + 1, "unrelated pool job failed");
        // The mapped-mailbox API is deliberately unprofiled and cannot leak stale phase_active_ state.
        std::memcpy(&mb[0], &inputs[0].mailbox, sizeof(sq::Mailbox)); mb[0].ready_seq = 177;
        cpu->begin_step(177, 0, 1); SQ_CHECK(cpu->end_step(), "legacy mailbox step failed");
        SQ_CHECK(std::memcmp(result[0].out[0], expected[0][0].data(), sizeof(float) * E) == 0, "legacy output changed");
        SQ_CHECK(cpu->phase_profile().seen_jobs == before.seen_jobs && cpu->phase_profile().sampled_jobs == before.sampled_jobs,
                 "unprofiled APIs changed event-layer diagnostics");
        cpu->stop(); cpu->stop();
    }
    sq::log("CPU PHASE PROFILE EXACT PASS: %llu float/guard checks; NT1..4, workers1/6, skip/limit/empty/partial/restart",
            (unsigned long long)checked);
    return 0;
}
