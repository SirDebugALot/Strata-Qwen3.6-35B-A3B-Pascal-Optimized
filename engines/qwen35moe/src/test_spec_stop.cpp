// GPU/model regression for stopping inside an accepted speculative window. Run only when the GPU is free.
// Usage: test_spec_stop model.gguf prompt.ids.txt [cache_mib=128] [threads=6] [mtp=3]
// A real accepted draft is temporarily designated EOS. This catches history overcommit without depending on
// a particular model's natural EOS position. Greedy, penalized, and stochastic top-k=1 paths are covered.
#include "common.hpp"
#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {
void env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

bool exact_logits(const std::vector<float>& expected, const std::vector<float>& actual) {
    if (expected.size() != actual.size() || expected.empty()) return false;
    for (size_t i = 0; i < expected.size(); ++i)
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) return false;
    return std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0;
}

struct AcceptedWindow {
    std::vector<int> prefix, output;
    int input = -1;
};

AcceptedWindow find_window(sq::Engine& engine, std::vector<int> prefix, const sq::SamplingParams& sampling) {
    // Restart each attempt from a prompt, just as a server request does. Failing to find an accepted draft is a
    // failed fixture, not a skipped/pass regression: the former bug requires an accepted token at the boundary.
    for (int attempt = 0; attempt < 24; ++attempt) {
        engine.reset();
        engine.feed(prefix.data(), (int)prefix.size());
        engine.begin_request(sampling);
        std::mt19937_64 rng(261008);
        const int input = engine.sample(sampling, rng);
        std::vector<int> output;
        engine.spec_step(input, sampling, rng, output);
        SQ_CHECK(!output.empty(), "unbounded step returned no output");
        if (output.size() > 1) return {prefix, output, input};
        prefix = engine.tokens();
    }
    sq::die("fixture produced no accepted drafts in 24 attempts; use a predictable language prompt");
}

void run_case(sq::Engine& engine, const AcceptedWindow& window, const sq::SamplingParams& sampling,
              const char* label, int stop_index, int limit, bool plain = false) {
    const int original_draft = engine.mtp_draft();
    engine.set_mtp_draft(plain ? 0 : original_draft);
    engine.reset();
    engine.feed(window.prefix.data(), (int)window.prefix.size());
    engine.begin_request(sampling);
    std::mt19937_64 rng(261008);
    const int input = engine.sample(sampling, rng);
    SQ_CHECK(input == window.input, "%s: initial token changed", label);
    std::vector<int> stops;
    if (stop_index >= 0) stops.push_back(window.output[(size_t)stop_index]);
    // Use a list containing an impossible token to also test the non-matching-stop-list path.
    else stops.push_back(-1);
    std::vector<int> expected;
    for (int token : window.output) {
        expected.push_back(token);
        if (plain || (int)expected.size() == limit || std::find(stops.begin(), stops.end(), token) != stops.end()) break;
    }
    const auto steps0 = engine.stats.decode_steps, accepted0 = engine.stats.accepted;
    std::vector<int> output;
    engine.spec_step(input, sampling, rng, output, &stops, limit);
    SQ_CHECK(output == expected, "%s: returned output differs from the unrestricted prefix", label);
    if (sampling.temperature > 0) {
        // With top_k=1 every distribution has one nonzero candidate. Both ordinary sampling and speculative
        // acceptance consume exactly one uniform draw per emitted token (including the initial input).
        // Sampling any un-emitted suffix past EOS would advance rng beyond this independent count.
        std::mt19937_64 expected_rng(261008);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        for (size_t i = 0; i <= output.size(); ++i) (void)uniform(expected_rng);
        SQ_CHECK(rng == expected_rng, "%s: RNG advanced past the returned output", label);
    }
    std::vector<int> committed = window.prefix;
    committed.push_back(input);
    committed.insert(committed.end(), output.begin(), output.end() - 1);
    SQ_CHECK(engine.tokens() == committed && engine.n_past() == (int)committed.size(),
             "%s: stopped step committed the final/un-emitted token(s)", label);
    SQ_CHECK(engine.stats.decode_steps - steps0 == output.size(), "%s: decode counter overcommitted", label);
    SQ_CHECK(engine.stats.accepted - accepted0 == output.size() - 1, "%s: accepted counter overcommitted", label);
    const auto stopped_logits = engine.logits_host();

    // An identical follow-up prompt plus its terminal token and a new user suffix must reuse the entire live
    // prefix with zero checkpoint slots. This is the server failure that originally exposed the bug.
    std::vector<int> followup = committed;
    followup.push_back(output.back());
    const int suffix[] = {198, 1000, 198};
    followup.insert(followup.end(), std::begin(suffix), std::end(suffix));
    const int reused = engine.reuse_prefix(followup);
    SQ_CHECK(reused == (int)committed.size(), "%s: follow-up reused %d of %zu live tokens", label, reused, committed.size());
    engine.feed(followup.data() + reused, (int)followup.size() - reused);
    const auto continued_logits = engine.logits_host();

    // Compare against ordinary token-by-token feeding of exactly the emitted prefix. Exact logits check both
    // recurrent-state rollback and stale future KV/MTP positions, beyond the visible history/vector assertions.
    engine.reset();
    engine.feed(committed.data(), (int)committed.size());
    SQ_CHECK(exact_logits(engine.logits_host(), stopped_logits), "%s: stopped state differs from serial feed", label);
    engine.feed(followup.data() + committed.size(), (int)(followup.size() - committed.size()));
    SQ_CHECK(exact_logits(engine.logits_host(), continued_logits), "%s: follow-up state differs from serial feed", label);
    engine.set_mtp_draft(original_draft);
    std::printf("%s candidate_sampler=%d stochastic=%d output=%zu committed=%zu reused=%d exact_logits=PASS\n",
                label, !sampling.greedy(), sampling.temperature > 0, output.size(), committed.size(), reused);
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) sq::die("usage: test_spec_stop model.gguf prompt.ids.txt [cache_mib=128] [threads=6] [mtp=3]");
    std::ifstream input(argv[2]);
    SQ_CHECK(input.good(), "cannot open prompt IDs: %s", argv[2]);
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream ids(text);
    std::vector<int> prompt;
    for (int token; ids >> token;) prompt.push_back(token);
    SQ_CHECK(ids.eof() && !prompt.empty() && prompt.size() <= 4096, "prompt must contain 1..4096 comma/space separated IDs");
    const int draft = argc > 5 ? std::atoi(argv[5]) : 3;
    SQ_CHECK(draft >= 1 && draft < sq::kMaxT, "mtp must be 1..3");
    env("STRATA_EVENT_MOE", "1");
    env("STRATA_SYNC_MOE", "0");
    env("STRATA_ASYNC_CACHE", "0");
    env("STRATA_CACHE_REBALANCE_EVERY", "0");
    env("STRATA_KV_INITIAL_TOKENS", "0");
    env("STRATA_PROMPT_BATCH", "1");
    sq::EngineOptions options;
    options.model_path = argv[1];
    options.ctx = (int)prompt.size() + 256;
    options.cache_mb = argc > 3 ? std::atoi(argv[3]) : 128;
    options.cpu_threads = argc > 4 ? std::atoi(argv[4]) : 6;
    options.vram_reserve_mb = 128;
    options.prefill_chunk = 0;
    options.mtp_draft = draft;
    options.draft_p = 0; // always verify the allocated window; no confidence-based skip can hide the regression
    options.ckpt_slots = 0;
    options.use_graph = false;
    options.adapt_every = 0; // freeze CPU/GPU placement for bit-exact arithmetic comparisons
    sq::Engine engine;
    std::string error;
    SQ_CHECK(engine.init(options, error), "%s", error.c_str());
    SQ_CHECK(engine.mtp(), "the model's MTP block was not loaded");
    for (int token : prompt) SQ_CHECK(token >= 0 && token < engine.cfg().n_vocab, "invalid prompt token %d", token);
    for (int candidate_sampler : {0, 1, 2}) {
        sq::SamplingParams sampling;
        if (candidate_sampler) {
            sampling.presence_penalty = 0.05f;
            sampling.frequency_penalty = 0.01f;
            sampling.penalty_last_n = 8; // exercise penalty-window evictions as well as the candidate path
        }
        if (candidate_sampler == 2) {
            sampling.temperature = 0.8f;
            sampling.top_k = 1; // deterministic token, but exercises real sampling/acceptance RNG consumption
        }
        const auto window = find_window(engine, prompt, sampling);
        std::printf("fixture sampling_mode=%d prefix=%zu accepted=%zu drafts=%d\n",
                    candidate_sampler, window.prefix.size(), window.output.size() - 1, draft);
        run_case(engine, window, sampling, "EOS-first-accepted", 0, INT_MAX);
        if (window.output.size() > 2)
            run_case(engine, window, sampling, "EOS-later-accepted", (int)window.output.size() - 2, INT_MAX);
        run_case(engine, window, sampling, "EOS-final-bonus", (int)window.output.size() - 1, INT_MAX);
        run_case(engine, window, sampling, "length-one", -1, 1);
        run_case(engine, window, sampling, "length-two", -1, 2);
        run_case(engine, window, sampling, "plain-stop", 0, 1, true);
    }
    std::puts("SPECULATIVE STOP / LENGTH / PREFIX REUSE PASS");
    return 0;
}
