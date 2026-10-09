// GPU probe; run only when the GPU is free. Compare separate full-reserve (initial=0) and lazy (initial=256)
// invocations with the same remaining arguments. The SQKV records must match, apart from normal FP tolerance.
// test_lazy_kv model.gguf output.bin [initial=256] [prompt_tokens=513] [mtp=0]
// Header: six uint32 {SQKV, version=1, vocabulary, max_context, int8_kv, mtp_drafts}.
// Records: three uint32 {kind, position, count}, then count float32 logits (kind1), or int32 token IDs (kind2).
// Capacity is printed, not stored in comparable records. This also exercises checkpoint reuse across growth.
#include "common.hpp"
#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <random>
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
}

int main(int argc, char** argv) {
    if (argc < 3) sq::die("usage: test_lazy_kv model.gguf output.bin [initial=256] [prompt_tokens=513] [mtp=0]");
    const int initial = argc > 3 ? std::atoi(argv[3]) : 256;
    const int n = argc > 4 ? std::atoi(argv[4]) : 513;
    const int mtp = argc > 5 ? std::atoi(argv[5]) : 0;
    SQ_CHECK(initial >= 0 && initial <= 25088 && n >= 128 && n <= 4096 && mtp >= 0 && mtp <= 3,
             "initial must be 0..25088, prompt_tokens 128..4096, mtp 0..3");
    env("STRATA_KV_INITIAL_TOKENS", std::to_string(initial).c_str());
    env("STRATA_EVENT_MOE", "1");
    env("STRATA_SYNC_MOE", "0");
    env("STRATA_ASYNC_CACHE", "0");
    env("STRATA_CACHE_REBALANCE_EVERY", "0");
    env("STRATA_PROMPT_BATCH", "1");
    env("STRATA_CPU_LM_HEAD", "1");
    sq::EngineOptions options;
    options.model_path = argv[1];
    options.ctx = 25000; // the user's full configured context remains available in both runs
    options.cache_mb = 0;
    options.adapt_every = 0;
    options.cpu_threads = 6;
    options.vram_reserve_mb = 128;
    options.prefill_chunk = 0;
    options.use_graph = false;
    options.mtp_draft = mtp;
    options.ckpt_slots = 1;
    sq::Engine engine;
    std::string error;
    if (!engine.init(options, error)) sq::die("%s", error.c_str());
    SQ_CHECK(engine.ctx() == 25088, "configured context changed to %d", engine.ctx());
    SQ_CHECK(mtp == 0 || engine.mtp(), "requested MTP block was not loaded");
    SQ_CHECK(engine.cache_slots() == 0, "growth parity requires zero expert cache slots");
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    SQ_CHECK(output.good(), "cannot open output %s", argv[2]);
    const uint32_t header[6] = {0x564b5153, 1, (uint32_t)engine.cfg().n_vocab, (uint32_t)engine.ctx(),
        std::string(engine.kv_type()) == "int8" ? 1u : 0u, (uint32_t)engine.mtp_draft()};
    output.write((const char*)header, sizeof(header));
    int snapshots = 0;
    auto snapshot = [&] {
        const auto logits = engine.logits_host();
        SQ_CHECK(std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }),
                 "nonfinite logits at position %d", engine.n_past());
        const uint32_t record[3] = {1, (uint32_t)engine.n_past(), (uint32_t)logits.size()};
        output.write((const char*)record, sizeof(record));
        output.write((const char*)logits.data(), logits.size() * sizeof(float));
        ++snapshots;
        sq::log("KV probe snapshot %d: position %d, capacity %d / %d, %.1f MiB",
                snapshots, engine.n_past(), engine.kv_capacity(), engine.ctx(), engine.kv_bytes() / 1048576.0);
    };
    std::mt19937 tokens_rng(261008);
    std::vector<int> tokens(n);
    const int range = std::min(50000, engine.cfg().n_vocab - 1000);
    for (int& token : tokens) token = 1000 + (int)(tokens_rng() % range);
    for (int i = 0; i < n; ++i) {
        engine.feed(tokens.data() + i, 1);
        const int p = i + 1;
        if (p == 128) engine.save_checkpoint();
        if (p == 128 || p == 255 || p == 256 || p == 257 || p == 511 || p == 512 || p == n) snapshot();
    }
    sq::SamplingParams sampling;
    sampling.temperature = 0;
    sampling.presence_penalty = sampling.frequency_penalty = 0;
    sampling.repetition_penalty = 1;
    engine.begin_request(sampling);
    std::mt19937_64 sampling_rng(99);
    int next = engine.sample(sampling, sampling_rng);
    for (int step = 0; step < 8; ++step) {
        std::vector<int> following;
        engine.spec_step(next, sampling, sampling_rng, following);
        SQ_CHECK(!following.empty(), "speculative step returned no continuation");
        const uint32_t record[3] = {2, (uint32_t)engine.n_past(), (uint32_t)following.size()};
        output.write((const char*)record, sizeof(record));
        output.write((const char*)following.data(), following.size() * sizeof(int));
        next = following.back();
        snapshot();
    }
    // Restore the earlier delta-net/MTP checkpoint while retaining the KV heads relocated during growth.
    std::vector<int> alternate = tokens;
    alternate.push_back(1234);
    const int reused = engine.reuse_prefix(alternate);
    SQ_CHECK(reused == 128, "expected checkpoint reuse of128 tokens, got %d", reused);
    engine.feed(alternate.data() + reused, (int)alternate.size() - reused);
    snapshot();
    SQ_CHECK(engine.ctx() == 25088 && engine.kv_capacity() >= engine.n_past(), "KV capacity/context invariant failed");
    output.close();
    SQ_CHECK(!output.fail(), "failed writing probe output");
    sq::log("KV PROBE COMPLETE: %d snapshots, initial %d, final capacity %d / %d, %s, MTP%d",
            snapshots, initial, engine.kv_capacity(), engine.ctx(), engine.kv_type(), engine.mtp_draft());
    return 0;
}
