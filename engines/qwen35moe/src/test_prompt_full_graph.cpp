// Exact accepted-T4 graph regression. Parent owns all GPU/model execution.
// One engine: actual graph runs first, then the uncaptured debug path at the SAME frozen cache placement.
// Exercises both head variants/replay counters, tails1..3, asynchronous replacements, MTP rollback,
// KV growth/recapture, changed expert slabs and an empty cache. No numerical tolerance fallback.
// Usage: test_prompt_full_graph model.gguf [cache_mib=128] [threads=6]
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
    std::vector<float> logits, rows, committed, continuation;
    std::vector<uint8_t> prefix, final_prefix;
    std::vector<int> history;
    uint64_t hits = 0, misses = 0, skipped_heads = 0, skipped_snapshots = 0;
};
State execute(sq::Engine& engine, int n, bool graph) {
    std::vector<std::vector<float>> debug;
    engine.debug_layers = graph ? nullptr : &debug;
    engine.reset();
    std::vector<int> tokens(n);
    for (int i = 0; i < n; ++i) tokens[i] = i % 5 == 0 ? 198 : 1000 + (i * 37) % 700;
    uint64_t h0 = 0, m0 = 0, h1 = 0, m1 = 0;
    engine.expert_counts(h0, m0);
    const uint64_t heads0 = engine.debug_prompt_heads_skipped();
    const uint64_t snapshots0 = engine.debug_prompt_snapshot_bytes_skipped();
    const uint64_t launches0 = engine.debug_prompt_full_graph_launches();
    const uint64_t intermediate0 = engine.debug_prompt_full_graph_variant_launches(true);
    const uint64_t final0 = engine.debug_prompt_full_graph_variant_launches(false);
    const uint64_t deferred0 = engine.debug_prompt_mtp_submitted();
    engine.feed(tokens.data(), n);
    State state;
    state.logits = engine.logits_host();
    state.prefix = engine.debug_prefix_state(); // includes the real pending count/hidden rows before normalization
    SQ_CHECK(engine.tokens() == tokens && engine.n_past() == n, "prompt history mismatch");
    state.skipped_heads = engine.debug_prompt_heads_skipped() - heads0;
    state.skipped_snapshots = engine.debug_prompt_snapshot_bytes_skipped() - snapshots0;
    const uint64_t expected = graph ? (uint64_t)n / 4 : 0;
    SQ_CHECK(engine.debug_prompt_full_graph_launches() - launches0 == expected,
             "T4 graph replay count mismatch (%d tokens, graph=%d)", n, graph);
    SQ_CHECK(engine.debug_prompt_full_graph_variant_launches(true) - intermediate0 ==
                 (graph ? (uint64_t)(n - 1) / 4 : 0), "intermediate graph count mismatch");
    SQ_CHECK(engine.debug_prompt_full_graph_variant_launches(false) - final0 ==
                 (graph && n % 4 == 0 ? 1u : 0u), "final-head graph count mismatch");
    SQ_CHECK(state.skipped_heads == (uint64_t)(n - 1) / 4 && state.skipped_snapshots > 0,
             "capture/replay prompt elision counters are incorrect or vacuous");
    if (graph) SQ_CHECK(engine.debug_prompt_mtp_submitted() > deferred0, "deferred MTP not exercised");

    // A prompt graph must never leak into speculative verification, which needs all logits and snapshots.
    debug.clear();
    const uint64_t before_verify = engine.debug_prompt_full_graph_launches();
    const int window[] = {1222, 1333, 1444, 1555};
    state.rows = engine.debug_verify(window, 4, 2);
    state.committed = engine.logits_host();
    SQ_CHECK(engine.debug_prompt_full_graph_launches() == before_verify, "verification used a prompt graph");
    tokens.insert(tokens.end(), window, window + 2);
    SQ_CHECK(engine.tokens() == tokens, "rollback history mismatch");
    const int next[] = {1666, 1777, 198};
    engine.feed(next, 3); // tail-only continuation; must remain uncaptured and catch up pending MTP pairs
    state.continuation = engine.logits_host();
    state.final_prefix = engine.debug_prefix_state();
    state.history = engine.tokens();
    tokens.insert(tokens.end(), next, next + 3);
    SQ_CHECK(state.history == tokens && engine.n_past() == (int)tokens.size(), "continuation history mismatch");
    SQ_CHECK(engine.debug_prompt_full_graph_launches() == before_verify, "tail continuation used a T4 graph");
    engine.expert_counts(h1, m1);
    state.hits = h1 - h0; state.misses = m1 - m0;
    engine.debug_layers = nullptr;
    return state;
}
void compare(sq::Engine& engine, const char* phase, int n, uint64_t expected_captures) {
    const uint64_t swaps = engine.stats.swaps;
    const int64_t slots = engine.cache_slots();
    const State actual = execute(engine, n, true);
    const State reference = execute(engine, n, false);
    SQ_CHECK(exact(actual.logits, reference.logits), "%s n%d prompt logits differ", phase, n);
    SQ_CHECK(exact(actual.rows, reference.rows) && exact(actual.committed, reference.committed),
             "%s n%d verification/rollback logits differ", phase, n);
    SQ_CHECK(exact(actual.continuation, reference.continuation), "%s n%d continuation logits differ", phase, n);
    SQ_CHECK(actual.prefix == reference.prefix && actual.final_prefix == reference.final_prefix,
             "%s n%d canonical recurrent/KV/pending state differs", phase, n);
    SQ_CHECK(actual.history == reference.history && actual.hits == reference.hits && actual.misses == reference.misses,
             "%s n%d history or route counters differ", phase, n);
    SQ_CHECK(actual.skipped_heads == reference.skipped_heads && actual.skipped_snapshots == reference.skipped_snapshots,
             "%s n%d replay bookkeeping differs from ordinary execution", phase, n);
    SQ_CHECK(engine.debug_prompt_full_graph_captures() == expected_captures,
             "%s n%d unexpected capture/recapture count", phase, n);
    SQ_CHECK(engine.stats.swaps == swaps && engine.cache_slots() == slots, "%s placement changed during comparison", phase);
    if (slots == 0) SQ_CHECK(actual.hits == 0 && actual.misses > 0, "empty cache route accounting failed");
    std::printf("%s n%d captures=%llu slots=%lld exact_logits/state/history/routes/counters=PASS\n",
                phase, n, (unsigned long long)expected_captures, (long long)slots);
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) sq::die("usage: test_prompt_full_graph model.gguf [cache_mib=128] [threads=6]");
    env("STRATA_EVENT_MOE", "1"); env("STRATA_SYNC_MOE", "0");
    env("STRATA_EVENT_MOE_GRAPH", "0"); env("STRATA_EVENT_GRAPH_BLOCKING", "0");
    env("STRATA_PROMPT_FULL_GRAPH", "1"); env("STRATA_PROMPT_ROUTE_GRAPH", "0");
    env("STRATA_PROMPT_SKIP_HEAD", "1"); env("STRATA_PROMPT_NO_SNAPSHOTS", "1");
    env("STRATA_PROMPT_ASYNC_MTP", "1"); env("STRATA_MTP_LAST_ROW", "1");
    env("STRATA_PROMPT_BATCH", "4"); env("STRATA_PROFILE_PIPELINE", "0");
    env("STRATA_CPU_PHASE_PROFILE", "0"); env("STRATA_TRACE_STEPS", "0");
    env("STRATA_MISS_STREAM_EXPERTS", "0"); env("STRATA_CACHE_REBALANCE_EVERY", "0");
    env("STRATA_ASYNC_CACHE", "1"); env("STRATA_KV_INITIAL_TOKENS", "256");
    env("STRATA_CPU_LM_HEAD", "0"); env("STRATA_NATIVE_DENSE_Q6", "0");
    env("STRATA_KV_INT8", "0"); env("STRATA_NATIVE_EXPERTS", "1");
    env("STRATA_PROMPT_COMPLETION_SPINS", "64");
    sq::EngineOptions options;
    options.model_path = argv[1]; options.ctx = 1024;
    options.cache_mb = argc > 2 ? std::atoi(argv[2]) : 128;
    options.cpu_threads = argc > 3 ? std::atoi(argv[3]) : 6;
    options.vram_reserve_mb = 128; options.prefill_chunk = 0;
    options.mtp_draft = 3; options.ckpt_slots = 0; options.use_graph = false;
    options.adapt_every = 4; options.adapt_swaps = 16; options.profile_weight = 0;
    sq::Engine engine; std::string error;
    SQ_CHECK(engine.init(options, error), "%s", error.c_str());
    SQ_CHECK(engine.mtp() && engine.kv_capacity() == 256 && engine.cache_slots() >= 4,
             "test requires MTP, lazy256 and a nonempty cache");
    const uint64_t swaps0 = engine.stats.swaps;
    const int warm[] = {198,1000,198,1001,198,1000,198,1001};
    for (int i = 0; i < 4 && engine.stats.swaps == swaps0; ++i) engine.feed(warm, 8);
    SQ_CHECK(engine.stats.swaps > swaps0 && engine.debug_prompt_full_graph_captures() == 2,
             "adaptive warmup did not exercise real swaps and both prompt graph variants");
    engine.debug_freeze_cache_adaptation(); // last upload stays pending until the actual graph replay waits for it
    for (int n : {8,13,14,15,129}) compare(engine, "retained-after-swaps", n, 2);
    SQ_CHECK(engine.debug_cache_payloads_match(), "adaptive cache bytes differ from model payload");

    engine.debug_reserve_kv(512); // ordinary drained growth; existing graph addresses/strides must invalidate
    SQ_CHECK(engine.kv_capacity() == 512 && engine.debug_prompt_full_graph_captures() == 2,
             "growth must defer recapture until first use");
    compare(engine, "KV-growth-recapture", 8, 4);
    compare(engine, "KV-growth-retained", 257, 4);
    const int64_t before = engine.cache_slots();
    engine.debug_trim_expert_cache((int64_t)(engine.cache_gib() * 1073741824.0) / 2);
    SQ_CHECK(engine.cache_slots() > 0 && engine.cache_slots() < before, "partial cache trim failed");
    compare(engine, "changed-cache-slabs", 8, 4);
    engine.debug_trim_expert_cache(std::numeric_limits<int64_t>::max());
    SQ_CHECK(engine.cache_slots() == 0, "full cache trim failed");
    compare(engine, "zero-resident-cache", 8, 4);
    std::puts("PROMPT FULL GRAPH EXACT REGRESSION PASS");
    return 0;
}
