// Exact model regression for the CPU-host-node graph. Parent serializes GPU/model runs.
// Usage: test_event_moe_graph model.gguf [cache_mib=128] [threads=6]
// One engine, frozen expert placement. debug_layers deliberately bypasses graphs for the reference execution.
#include "engine.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

bool exact(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct State {
    std::vector<float> rows, logits, continuation;
    std::vector<uint8_t> prefix;
    std::vector<int> history;
};

State execute(sq::Engine& engine, int T, int keep, bool graph) {
    const int prefix[] = {198, 1000, 198, 1011};
    const int window[] = {1222, 1333, 1444, 1555};
    const int continuation[] = {1666, 198};
    std::vector<std::vector<float>> debug;
    engine.debug_layers = graph ? nullptr : &debug;
    engine.reset();
    engine.feed(prefix, 4);
    debug.clear();
    State state;
    if (T == 1) {
        engine.feed(window, 1);
        state.rows = engine.logits_host();
    } else {
        state.rows = engine.debug_verify(window, T, keep);
    }
    debug.clear();
    state.logits = engine.logits_host();
    state.prefix = engine.debug_prefix_state();
    state.history = engine.tokens();
    engine.feed(continuation, 2);
    debug.clear();
    state.continuation = engine.logits_host();
    engine.debug_layers = nullptr;
    return state;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) sq::die("usage: test_event_moe_graph model.gguf [cache_mib=128] [threads=6]");
    env("STRATA_EVENT_MOE", "1");
    env("STRATA_SYNC_MOE", "0");
    env("STRATA_EVENT_MOE_GRAPH", "1");
    env("STRATA_PROFILE_PIPELINE", "0");
    env("STRATA_TRACE_STEPS", "0");
    env("STRATA_ASYNC_CACHE", "1");
    env("STRATA_CACHE_REBALANCE_EVERY", "0");
    env("STRATA_PROMPT_BATCH", "1");
    env("STRATA_KV_INITIAL_TOKENS", "256");
    sq::EngineOptions options;
    options.model_path = argv[1];
    options.ctx = 512;
    options.cache_mb = argc > 2 ? std::atoi(argv[2]) : 128;
    options.cpu_threads = argc > 3 ? std::atoi(argv[3]) : 6;
    options.vram_reserve_mb = 128;
    options.prefill_chunk = 0;
    options.mtp_draft = 3;
    options.ckpt_slots = 0;
    options.use_graph = false; // explicitly selected event graphs are independent of legacy --no-graph
    options.adapt_every = 0;
    sq::Engine engine;
    std::string error;
    SQ_CHECK(engine.init(options, error), "%s", error.c_str());
    SQ_CHECK(engine.mtp(), "test requires the model's MTP block");
    SQ_CHECK(engine.kv_capacity() == 256, "test requires initial 256-token lazy KV");
    const int growth_prefix[] = {198, 1000, 198, 1011};
    engine.feed(growth_prefix, 4);
    SQ_CHECK(engine.event_graph_steps() > 0, "trunk graph was never launched before KV growth");
    {
        const auto before = engine.debug_prefix_state();
        engine.debug_reserve_kv(512);
        SQ_CHECK(engine.kv_capacity() == 512 && engine.debug_prefix_state() == before,
                 "KV growth changed the live prefix before recapture");
    }
    // Growth can trim expert slots. Every reference/graph pair below starts AFTER that trim with frozen
    // placement, so CPU/GPU activation quantization changes cannot masquerade as a state-preservation bug.
    for (int T = 1; T <= 4; ++T) {
        for (int keep = 1; keep <= T; keep += std::max(1, T - 1)) {
            const uint64_t before = engine.event_graph_steps();
            const State reference = execute(engine, T, keep, false);
            SQ_CHECK(engine.event_graph_steps() == before, "reference unexpectedly used an event graph");
            const State actual = execute(engine, T, keep, true);
            SQ_CHECK(engine.event_graph_steps() > before, "candidate silently bypassed event graphs");
            SQ_CHECK(exact(reference.rows, actual.rows), "T=%d keep=%d verification rows differ", T, keep);
            SQ_CHECK(exact(reference.logits, actual.logits), "T=%d keep=%d committed logits differ", T, keep);
            SQ_CHECK(reference.prefix == actual.prefix, "T=%d keep=%d recurrent/KV/pending prefix differs", T, keep);
            SQ_CHECK(reference.history == actual.history, "T=%d keep=%d history differs", T, keep);
            SQ_CHECK(exact(reference.continuation, actual.continuation), "T=%d keep=%d continuation differs", T, keep);
            std::printf("T=%d keep=%d graph_launches=%llu exact_rows/state/continuation=PASS\n", T, keep,
                        (unsigned long long)(engine.event_graph_steps() - before));
        }
    }
    std::puts("EVENT MOE GRAPH / EXACT TRUNK / ROLLBACK / LAZY KV RECAPTURE PASS");
    return 0;
}
