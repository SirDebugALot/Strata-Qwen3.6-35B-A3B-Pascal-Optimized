// Retained-graph regression: real asynchronous swaps, changed cache slab capacity, then zero GPU experts.
// Every exact comparison freezes placement first. The graph candidate runs before the uncaptured reference,
// so the first replay must itself honor the pending upload dependency. Parent serializes all GPU/model runs.
// Usage: test_event_graph_cache model.gguf [cache_mib=128] [threads=6]
// STRATA_EVENT_GRAPH_BLOCKING is inherited, allowing separate process runs for both completion modes.
// STRATA_CACHE_DIRECT_PINNED is inherited; actual DMA use and byte-identical cache payloads are checked.
#include "engine.hpp"
#include "common.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    if (a.empty() || a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct State {
    std::vector<float> rows, logits, continuation;
    std::vector<uint8_t> prefix;
    std::vector<int> history;
};

State execute(sq::Engine& engine, int T, bool graph) {
    const int prefix[] = {198, 1000, 198};
    const int window[] = {1222, 1333, 1444, 1555};
    const int continuation[] = {1666, 198};
    std::vector<std::vector<float>> debug;
    engine.debug_layers = graph ? nullptr : &debug;
    engine.reset();
    engine.feed(prefix, 3);
    debug.clear();
    State state;
    if (T == 1) {
        engine.feed(window, 1);
        state.rows = engine.logits_host();
    } else {
        state.rows = engine.debug_verify(window, T, 1); // reject the later positions to exercise rollback/pending MTP
    }
    debug.clear();
    state.logits = engine.logits_host();
    state.prefix = engine.debug_prefix_state();
    state.history = engine.tokens();
    engine.feed(continuation, 2);
    state.continuation = engine.logits_host();
    engine.debug_layers = nullptr;
    return state;
}

void compare(sq::Engine& engine, const char* phase, uint64_t captures) {
    const uint64_t swaps = engine.stats.swaps;
    const int64_t slots = engine.cache_slots();
    uint64_t hits_before = 0, misses_before = 0;
    if (slots == 0) engine.expert_counts(hits_before, misses_before);
    for (int T : {1, 4}) {
        const uint64_t before = engine.event_graph_steps();
        const State actual = execute(engine, T, true); // intentionally before reference; pending uploads may remain
        const uint64_t after = engine.event_graph_steps();
        SQ_CHECK(after > before, "%s T=%d graph candidate bypassed graphs", phase, T);
        const State reference = execute(engine, T, false);
        SQ_CHECK(engine.event_graph_steps() == after, "%s T=%d reference unexpectedly used a graph", phase, T);
        SQ_CHECK(engine.debug_event_graph_captures() == captures,
                 "%s T=%d recaptured instead of reusing the existing graph", phase, T);
        SQ_CHECK(engine.stats.swaps == swaps && engine.cache_slots() == slots,
                 "%s T=%d cache placement changed during exact comparison", phase, T);
        SQ_CHECK(exact(reference.rows, actual.rows), "%s T=%d verification rows differ", phase, T);
        SQ_CHECK(exact(reference.logits, actual.logits), "%s T=%d committed logits differ", phase, T);
        SQ_CHECK(reference.prefix == actual.prefix, "%s T=%d recurrent/KV/pending prefix differs", phase, T);
        SQ_CHECK(reference.history == actual.history, "%s T=%d history differs", phase, T);
        SQ_CHECK(exact(reference.continuation, actual.continuation), "%s T=%d continuation differs", phase, T);
        std::printf("%s T=%d retained_captures=%llu slots=%lld graph_launches=%llu exact_rows/state/continuation=PASS\n",
            phase, T, (unsigned long long)captures, (long long)slots, (unsigned long long)(after - before));
    }
    if (slots == 0) {
        uint64_t hits_after = 0, misses_after = 0;
        engine.expert_counts(hits_after, misses_after);
        SQ_CHECK(hits_after == hits_before && misses_after > misses_before,
                 "empty cache unexpectedly routed resident experts or skipped all CPU experts");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) sq::die("usage: test_event_graph_cache model.gguf [cache_mib=128] [threads=6]");
    env("STRATA_EVENT_MOE", "1");
    env("STRATA_SYNC_MOE", "0");
    env("STRATA_EVENT_MOE_GRAPH", "1");
    env("STRATA_PROFILE_PIPELINE", "0");
    env("STRATA_TRACE_STEPS", "0");
    env("STRATA_ASYNC_CACHE", "1");
    env("STRATA_CACHE_REBALANCE_EVERY", "0");
    env("STRATA_PROMPT_BATCH", "1");
    env("STRATA_KV_INITIAL_TOKENS", "0"); // no KV growth/graph invalidation can conceal a stale cache pointer
    sq::EngineOptions options;
    options.model_path = argv[1];
    options.ctx = 256;
    options.cache_mb = argc > 2 ? std::atoi(argv[2]) : 128;
    options.cpu_threads = argc > 3 ? std::atoi(argv[3]) : 6;
    options.vram_reserve_mb = 128;
    options.prefill_chunk = 0;
    options.mtp_draft = 3;
    options.ckpt_slots = 0;
    options.use_graph = false;
    options.adapt_every = 1;
    options.adapt_swaps = 16;
    options.profile_weight = 0;
    sq::Engine engine;
    std::string error;
    SQ_CHECK(engine.init(options, error), "%s", error.c_str());
    SQ_CHECK(engine.mtp() && engine.cache_slots() >= 4, "test needs MTP and at least four resident expert slots");
    const uint64_t captures = engine.debug_event_graph_captures();
    SQ_CHECK(captures == 4, "test expects all four trunk graph variants to be captured once");
    const uint64_t old_swaps = engine.stats.swaps;
    const uint64_t old_direct = engine.debug_direct_pinned_uploads();
    // Repeated text builds live frequency above the normal swap hysteresis. Stop immediately after an actual
    // upload round, before another graph/reference invocation can finish that round incidentally.
    for (int i = 0; i < 24 && engine.stats.swaps == old_swaps; ++i) {
        const int token = i % 3 == 0 ? 198 : 1000;
        engine.feed(&token, 1);
    }
    SQ_CHECK(engine.stats.swaps > old_swaps, "no actual asynchronous expert replacement occurred");
    SQ_CHECK(engine.debug_event_graph_captures() == captures && engine.event_graph_steps() > 0,
             "graphs were replaced or bypassed during adaptive warmup");
    engine.debug_freeze_cache_adaptation(); // deliberately does not wait for the queued upload round
    std::printf("actual async swaps=%llu; preserving capture count=%llu\n",
        (unsigned long long)(engine.stats.swaps - old_swaps), (unsigned long long)captures);
    compare(engine, "after-async-swaps", captures);
    // Run only AFTER the pending-upload graph replay, so this readback cannot hide a missing graph dependency.
    const char* direct_env = std::getenv("STRATA_CACHE_DIRECT_PINNED");
    const bool direct_enabled = direct_env && std::strcmp(direct_env, "1") == 0;
    bool all_pinned = true, any_pinned = false;
    for (const auto& layer : engine.model().layers) {
        all_pinned = all_pinned && layer.experts_pinned;
        any_pinned = any_pinned || layer.experts_pinned;
    }
    const uint64_t direct_uploads = engine.debug_direct_pinned_uploads() - old_direct;
    if (direct_enabled && all_pinned)
        SQ_CHECK(direct_uploads == engine.stats.swaps - old_swaps && direct_uploads > 0,
                 "direct pinned mode did not upload every actual replacement directly");
    if (!direct_enabled || !any_pinned)
        SQ_CHECK(direct_uploads == 0, "staged/pageable mode unexpectedly used a direct pinned source");
    SQ_CHECK(engine.debug_cache_payloads_match(), "resident cache payload differs from immutable model weights");
    std::printf("cache upload source=%s actual_direct_uploads=%llu all_resident_bytes=PASS\n",
        direct_uploads ? "direct-pinned" : "staged", (unsigned long long)direct_uploads);

    const int64_t original_slots = engine.cache_slots();
    const int64_t trim_bytes = (int64_t)(engine.cache_gib() * 1073741824.0) / 2;
    engine.debug_trim_expert_cache(trim_bytes); // normal drain/reallocate/upload path, without graph invalidation
    SQ_CHECK(engine.cache_slots() > 0 && engine.cache_slots() < original_slots,
             "partial trim did not leave a smaller nonempty resident cache");
    SQ_CHECK(engine.debug_event_graph_captures() == captures, "partial trim invalidated the retained graphs");
    compare(engine, "after-slab-resize", captures);

    engine.debug_trim_expert_cache(std::numeric_limits<int64_t>::max());
    SQ_CHECK(engine.cache_slots() == 0, "full trim did not empty every GPU expert slab");
    SQ_CHECK(engine.debug_event_graph_captures() == captures, "full trim invalidated the retained graphs");
    compare(engine, "after-zero-hit-cache", captures);
    std::puts("RETAINED EVENT GRAPH / ASYNC SWAPS / CHANGED SLABS / ZERO HIT CACHE PASS");
    return 0;
}
