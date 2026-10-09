// Bounded full-context KV pressure test; parent process must serialize GPU use and impose a timeout.
// test_lazy_kv_pressure model.gguf [profile|-] [reserve_mib=32] [extra_mib=384] [int8_kv=0]
// Uses native experts, GPU LM head, MTP3, active auto-sized cache, and real asynchronous adaptation.
// 96 training + 32 prefix tokens trigger an upload exactly before growth; fewer than another 128 fed
// tokens keep the resulting cache placement fixed for continuation from the saved checkpoint.
// Recomputing the prefix after trimming is deliberately excluded: its CPU/GPU expert placement changes
// and changes activation rounding. The original test used that invalid reference and failed; this version
// checks canonical live state byte-for-byte during relocation, then replays the identical saved state.
#include "common.hpp"
#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int kPrefix = 32;
void env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
int integer(const char* value, int hi) {
    char* end = nullptr;
    const long result = std::strtol(value, &end, 10);
    SQ_CHECK(value[0] && end && !*end && result >= 0 && result <= hi, "invalid integer: %s", value);
    return (int)result;
}
int64_t cache_bytes(const sq::Engine& engine) {
    return (int64_t)std::llround(engine.cache_gib() * 1073741824.0);
}
void memory(const sq::Engine& engine, const char* phase) {
    size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    std::printf("MEMORY phase=%s pos=%d capacity=%d max=%d kv_bytes=%lld cache_bytes=%lld slots=%lld "
                "cuda_free_bytes=%llu cuda_total_bytes=%llu\n", phase, engine.n_past(), engine.kv_capacity(),
                engine.ctx(), (long long)engine.kv_bytes(), (long long)cache_bytes(engine),
                (long long)engine.cache_slots(), (unsigned long long)free, (unsigned long long)total);
    std::fflush(stdout);
}
void finite(const std::vector<float>& values, const char* label) {
    SQ_CHECK(!values.empty(), "%s is empty", label);
    double magnitude = 0;
    for (float value : values) {
        SQ_CHECK(std::isfinite(value), "%s contains nonfinite values", label);
        magnitude += std::abs((double)value);
    }
    SQ_CHECK(magnitude > 0, "%s contains only zeroes", label);
}
void history(const sq::Engine& engine, const std::vector<int>& expected) {
    SQ_CHECK(engine.n_past() == (int)expected.size() && engine.tokens() == expected,
             "history/position changed unexpectedly: %d versus %zu", engine.n_past(), expected.size());
}
struct Trace {
    std::vector<std::vector<float>> logits;
    std::vector<int> continuation;
    std::vector<int> history;
};
Trace continuation(sq::Engine& engine, const std::vector<int>& prefix, const std::vector<int>& tail) {
    Trace trace;
    std::vector<int> expected = prefix;
    // Force a rejected verification suffix, then overwrite it with a different continuation.
    const auto verified = engine.debug_verify(tail.data(), 4, 1);
    const size_t vocab = engine.cfg().n_vocab;
    SQ_CHECK(verified.size() == 4 * vocab, "verification returned wrong number of rows");
    for (int row = 0; row < 4; ++row) {
        trace.logits.emplace_back(verified.begin() + row * vocab, verified.begin() + (row + 1) * vocab);
        finite(trace.logits.back(), "verification logits");
    }
    expected.push_back(tail[0]);
    history(engine, expected);
    for (int i = 4; i < 8; ++i) {
        engine.feed(tail.data() + i, 1);
        expected.push_back(tail[i]);
        history(engine, expected);
        trace.logits.push_back(engine.logits_host());
        finite(trace.logits.back(), "post-rollback logits");
    }
    sq::SamplingParams sampling;
    sampling.temperature = 0;
    engine.begin_request(sampling);
    std::mt19937_64 rng(9918);
    const int next = engine.sample(sampling, rng);
    const auto drafted = engine.stats.drafted;
    engine.spec_step(next, sampling, rng, trace.continuation);
    SQ_CHECK(!trace.continuation.empty() && engine.stats.drafted - drafted == 3,
             "MTP3 did not execute its full forced draft window");
    expected.push_back(next);
    expected.insert(expected.end(), trace.continuation.begin(), trace.continuation.end() - 1);
    history(engine, expected);
    trace.history = engine.tokens();
    trace.logits.push_back(engine.logits_host());
    finite(trace.logits.back(), "MTP continuation logits");
    return trace;
}
bool compare(const std::vector<float>& reference, const std::vector<float>& actual, int row) {
    SQ_CHECK(reference.size() == actual.size(), "comparison size mismatch");
    double mr = -INFINITY, ma = -INFINITY, sr = 0, sa = 0;
    size_t ar = 0, aa = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        mr = std::max(mr, (double)reference[i]);
        ma = std::max(ma, (double)actual[i]);
        if (reference[i] > reference[ar]) ar = i;
        if (actual[i] > actual[aa]) aa = i;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
        sr += std::exp(reference[i] - mr);
        sa += std::exp(actual[i] - ma);
    }
    double kl = 0, max_abs = 0, active_logp = 0;
    const double zr = mr + std::log(sr), za = ma + std::log(sa);
    for (size_t i = 0; i < reference.size(); ++i) {
        const double lr = reference[i] - zr, la = actual[i] - za;
        kl += std::exp(lr) * (lr - la);
        max_abs = std::max(max_abs, std::abs((double)reference[i] - actual[i]));
        if (lr > -10) active_logp = std::max(active_logp, std::abs(lr - la));
    }
    const bool exact = std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)) == 0;
    const bool ok = exact && std::isfinite(kl) && kl <= 1e-4 && active_logp <= 0.10 && ar == aa;
    std::printf("PARITY row=%d bitwise=%d KL=%.9g max_dlogit=%.9g active_dlogp=%.9g top1=%zu/%zu %s\n",
                row, exact ? 1 : 0, kl, max_abs, active_logp, ar, aa, ok ? "PASS" : "FAIL");
    return ok;
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 6) {
        std::fprintf(stderr, "usage: test_lazy_kv_pressure model.gguf [profile|-] [reserve_mib=32] [extra_mib=384] [int8_kv=0]\n");
        return 2;
    }
    const int reserve = argc > 3 ? integer(argv[3], 4096) : 32;
    const int extra = argc > 4 ? integer(argv[4], 1024) : 384;
    const int int8 = argc > 5 ? integer(argv[5], 1) : 0;
    env("STRATA_NATIVE_EXPERTS", "1");
    env("STRATA_CPU_LM_HEAD", "0");
    env("STRATA_GPU_EXPERT_REUSE", "0"); // isolate KV relocation from separately tested kernel reuse
    env("STRATA_KV_INITIAL_TOKENS", "2048");
    env("STRATA_KV_INT8", std::to_string(int8).c_str());
    env("STRATA_EVENT_MOE", "1");
    env("STRATA_SYNC_MOE", "0");
    env("STRATA_EVENT_BLOCKING", "0");
    env("STRATA_ASYNC_CACHE", "1");
    env("STRATA_CACHE_REBALANCE_EVERY", "0");
    env("STRATA_PROMPT_BATCH", "4");
    env("STRATA_CACHE_INIT_RESERVE_MIB", "0");
    env("STRATA_CACHE_BUDGET_EXTRA_MIB", std::to_string(extra).c_str());
    sq::EngineOptions options;
    options.model_path = argv[1];
    if (argc > 2 && std::strcmp(argv[2], "-") != 0) options.profile_path = argv[2];
    options.ctx = 25000;
    options.cache_mb = -1;
    options.vram_reserve_mb = reserve;
    options.cpu_threads = 6;
    options.prefill_chunk = 0;
    options.use_graph = false;
    options.mtp_draft = 3;
    options.draft_p = 0; // execute all three drafts rather than skipping low-probability drafts
    options.ckpt_slots = 1;
    options.adapt_every = 128;
    options.adapt_swaps = 16;
    options.adapt_decay = 0.70;
    sq::Engine engine;
    std::string error;
    if (!engine.init(options, error)) sq::die("%s", error.c_str());
    SQ_CHECK(engine.ctx() == 25088 && engine.kv_capacity() == 2048 && engine.mtp_draft() == 3 &&
             !engine.model().cpu_lm_head && engine.cache_slots() > 0, "initial pressure configuration mismatch");
    memory(engine, "initial");
    std::mt19937 rng(261008);
    const int range = std::min(50000, engine.cfg().n_vocab - 1000);
    SQ_CHECK(range > 0, "unexpected vocabulary size");
    auto tokens = [&](int count) {
        std::vector<int> result(count);
        for (int& token : result) token = 1000 + (int)(rng() % range);
        return result;
    };
    const auto training = tokens(96), prefix = tokens(kPrefix), tail = tokens(8);
    engine.feed(training.data(), (int)training.size());
    engine.reset();
    engine.feed(prefix.data(), (int)prefix.size());
    history(engine, prefix);
    SQ_CHECK(engine.stats.swaps > 0, "training did not enqueue an asynchronous expert upload");
    const auto fixed_swaps = engine.stats.swaps;
    const auto original_logits = engine.logits_host(), original_hidden = engine.debug_hidden();
    finite(original_logits, "nonzero prefix logits");
    finite(original_hidden, "nonzero prefix hidden state");
    engine.save_checkpoint();
    const auto original_state = engine.debug_prefix_state();
    SQ_CHECK(original_state.size() > 12 && std::any_of(original_state.begin() + 12, original_state.end(),
             [](uint8_t byte) { return byte != 0; }), "canonical prefix state has no nonzero payload");
    const int64_t original_cache = cache_bytes(engine), original_kv = engine.kv_bytes();
    for (int target : {4096, 8192, 16384, 25088}) {
        const int64_t before_cache = cache_bytes(engine), before_kv = engine.kv_bytes();
        const int64_t before_slots = engine.cache_slots();
        engine.debug_reserve_kv(target);
        const int64_t delta = engine.kv_bytes() - before_kv, released = before_cache - cache_bytes(engine);
        SQ_CHECK(engine.kv_capacity() == target && engine.ctx() == 25088 && delta > 0,
                 "KV growth/capacity invariant failed");
        SQ_CHECK(released >= std::min(delta, before_cache) && engine.cache_slots() < before_slots,
                 "cache did not fund KV growth: released %lld, KV delta %lld", (long long)released, (long long)delta);
        history(engine, prefix);
        const auto after_logits = engine.logits_host(), after_hidden = engine.debug_hidden();
        SQ_CHECK(after_logits == original_logits && after_hidden == original_hidden,
                 "growth modified existing logits or residual state");
        SQ_CHECK(engine.debug_prefix_state() == original_state,
                 "growth modified canonical recurrent/pending-MTP/KV prefix bytes at capacity %d", target);
        std::printf("PREFIX_STATE target=%d bytes=%zu bitwise=1\n", target, original_state.size());
        std::printf("GROWTH target=%d kv_delta=%lld cache_released=%lld\n", target, (long long)delta, (long long)released);
        memory(engine, "grown");
    }
    SQ_CHECK(original_cache - cache_bytes(engine) >= engine.kv_bytes() - original_kv && engine.cache_slots() > 0,
             "full-context growth failed to retain a funded active expert cache");
    // Both continuations start from the very same recurrent, MTP-hidden and causal KV prefix state, with
    // identical post-trim cache placement. Restore the checkpoint instead of recomputing the prefix.
    const Trace preserved = continuation(engine, prefix, tail);
    auto restore_prompt = prefix;
    restore_prompt.push_back(tail[0]);
    SQ_CHECK(engine.reuse_prefix(restore_prompt) == kPrefix, "failed to restore pre-growth checkpoint");
    history(engine, prefix);
    SQ_CHECK(engine.debug_prefix_state() == original_state, "checkpoint did not restore the identical live prefix state");
    const Trace replayed = continuation(engine, prefix, tail);
    SQ_CHECK(engine.stats.swaps == fixed_swaps, "cache changed during the preservation comparison");
    bool ok = preserved.continuation == replayed.continuation && preserved.history == replayed.history;
    SQ_CHECK(preserved.logits.size() == replayed.logits.size(), "trace length mismatch");
    for (size_t row = 0; row < preserved.logits.size(); ++row)
        ok = compare(replayed.logits[row], preserved.logits[row], (int)row) && ok;
    std::printf("MTP_HISTORY %s\n", preserved.history == replayed.history &&
                preserved.continuation == replayed.continuation ? "PASS" : "FAIL");
    memory(engine, "complete");
    std::printf("KV PRESSURE %s: initial=2048 final=%d context=%d KV=%s MTP3 reserve=%d extra=%d swaps=%llu\n",
                ok ? "PASS" : "FAIL", engine.kv_capacity(), engine.ctx(), engine.kv_type(), reserve, extra,
                (unsigned long long)fixed_swaps);
    return ok ? 0 : 1;
}
