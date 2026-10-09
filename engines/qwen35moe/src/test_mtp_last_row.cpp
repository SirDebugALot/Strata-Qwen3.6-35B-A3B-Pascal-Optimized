// Full-model parity for MTP pending-pair compaction. Parent serializes all GPU/model runs.
// Usage: test_mtp_last_row model.gguf [cache_mib=128] [threads=6]
// One engine, fixed cache placement; compares original full-row execution with the opt-in path.
#include "engine.hpp"
#include "common.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
    std::vector<float> logits;
    std::vector<uint8_t> after_catch_up, final_prefix;
    std::vector<std::vector<int>> histories, outputs;
    std::mt19937_64 rng;
};

State execute(sq::Engine& engine, int prefix_tokens, int pending, bool need_draft, bool optimized) {
    engine.debug_set_mtp_last_row(optimized);
    engine.reset();
    std::vector<int> prefix(prefix_tokens);
    for (int i = 0; i < prefix_tokens; ++i) prefix[i] = i % 3 == 0 ? 198 : 1000 + i % 17;
    const int window[] = {1222, 1333, 1444, 1555};
    const int continuation[] = {1666, 198, 1777};
    engine.feed(prefix.data(), (int)prefix.size()); // also exercises state-only catch-up during prompting
    State state;
    auto save_logits = [&](const std::vector<float>& row) {
        state.logits.insert(state.logits.end(), row.begin(), row.end());
    };
    // Verify four target rows, then reject a suffix to leave exactly 1..4 pending MTP pairs. Future KV
    // positions from the rejected suffix must not affect either the MTP catch-up or its continuation.
    save_logits(engine.debug_verify(window, 4, pending));
    save_logits(engine.logits_host());
    state.histories.push_back(engine.tokens());
    const uint64_t tail0 = engine.debug_mtp_last_row_steps(), kv0 = engine.debug_mtp_kv_only_steps();
    save_logits(engine.debug_mtp_catch_up(continuation[0], need_draft));
    if (optimized && need_draft && pending > 1)
        SQ_CHECK(engine.debug_mtp_last_row_steps() == tail0 + 1, "last-row catch-up was bypassed");
    if (optimized && !need_draft)
        SQ_CHECK(engine.debug_mtp_kv_only_steps() == kv0 + 1, "KV-only catch-up was bypassed");
    if (!optimized)
        SQ_CHECK(engine.debug_mtp_last_row_steps() == tail0 && engine.debug_mtp_kv_only_steps() == kv0,
                 "reference unexpectedly used optimized catch-up");
    state.after_catch_up = engine.debug_prefix_state();
    save_logits(engine.logits_host()); // MTP must not overwrite the target distribution
    engine.feed(continuation, 3);
    save_logits(engine.logits_host());
    state.histories.push_back(engine.tokens());

    // Real drafting/verification after the explicit state comparison, including a one-token output limit.
    // Top-k=1 with penalties also checks the candidate-sampling path and exact RNG progression.
    sq::SamplingParams sampling;
    sampling.temperature = need_draft ? 0.0f : 0.8f;
    sampling.top_k = 1;
    sampling.top_p = 1.0f;
    sampling.min_p = 0;
    sampling.repetition_penalty = need_draft ? 1.0f : 1.05f;
    engine.begin_request(sampling);
    state.rng.seed(261008);
    int next = engine.sample(sampling, state.rng);
    const std::vector<int> no_stop = {-1};
    for (int limit : {4, 1, 4}) {
        std::vector<int> output;
        engine.spec_step(next, sampling, state.rng, output, &no_stop, limit);
        SQ_CHECK(!output.empty() && (int)output.size() <= limit, "invalid bounded speculative output");
        next = output.back();
        state.outputs.push_back(output);
        state.histories.push_back(engine.tokens());
        save_logits(engine.logits_host());
    }
    state.final_prefix = engine.debug_prefix_state();
    return state;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) sq::die("usage: test_mtp_last_row model.gguf [cache_mib=128] [threads=6]");
    env("STRATA_EVENT_MOE", "1");
    env("STRATA_SYNC_MOE", "0");
    env("STRATA_MTP_LAST_ROW", "1"); // allocate persistent scratch; each reference temporarily disables it
    env("STRATA_PROFILE_PIPELINE", "0");
    env("STRATA_TRACE_STEPS", "0");
    env("STRATA_CACHE_REBALANCE_EVERY", "0");
    env("STRATA_PROMPT_BATCH", "1");
    env("STRATA_KV_INITIAL_TOKENS", "0");
    sq::EngineOptions options;
    options.model_path = argv[1];
    options.ctx = 256;
    options.cache_mb = argc > 2 ? std::atoi(argv[2]) : 128;
    options.cpu_threads = argc > 3 ? std::atoi(argv[3]) : 6;
    options.vram_reserve_mb = 128;
    options.prefill_chunk = 0;
    options.mtp_draft = 3;
    options.draft_p = 0; // exercise the complete draft depth instead of confidence-dependent bypasses
    options.ckpt_slots = 0;
    options.use_graph = false;
    options.adapt_every = 0; // removing unused MTP routes must not change placement during this exact comparison
    sq::Engine engine;
    std::string error;
    SQ_CHECK(engine.init(options, error), "%s", error.c_str());
    SQ_CHECK(engine.mtp(), "test requires one full-attention MTP layer");
    for (int prefix_tokens : {3, 129}) {
      for (int pending = 1; pending <= 4; ++pending) {
        // The extra long-context fixture crosses two 64-key attention chunks, verifies four target rows,
        // then retains two. Both drafted and KV-only catch-up must ignore its rejected future KV suffix.
        if (prefix_tokens == 129 && pending != 2) continue;
        for (bool need_draft : {false, true}) {
            const State reference = execute(engine, prefix_tokens, pending, need_draft, false);
            const State actual = execute(engine, prefix_tokens, pending, need_draft, true);
            SQ_CHECK(exact(reference.logits, actual.logits), "pending=%d draft=%d full logits differ", pending, need_draft);
            SQ_CHECK(reference.after_catch_up == actual.after_catch_up,
                     "pending=%d draft=%d canonical KV/recurrent state differs after catch-up", pending, need_draft);
            SQ_CHECK(reference.final_prefix == actual.final_prefix,
                     "pending=%d draft=%d continuation state differs", pending, need_draft);
            SQ_CHECK(reference.histories == actual.histories && reference.outputs == actual.outputs && reference.rng == actual.rng,
                     "pending=%d draft=%d history/output/RNG differs", pending, need_draft);
            std::printf("prefix=%d pending=%d draft=%d exact_all_logits/KV/history/rollback/continuation/RNG=PASS\n",
                        prefix_tokens, pending, need_draft);
        }
      }
    }
    std::puts("MTP LAST ROW / KV-ONLY / PENDING 1..4 / EXACT MODEL PARITY PASS");
    return 0;
}
