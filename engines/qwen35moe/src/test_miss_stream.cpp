// Fixed-placement transient expert DMA regression. Parent serializes GPU/model runs.
// Usage: test_miss_stream model.gguf [eligible_limit=2] [threads=6] [cache_mib=0]
// This compares asynchronous and serialized execution of the SAME CPU/GPU placement. Moving original CPU
// experts to GPU changes activation rounding; model-quality validation against the original reference is separate.
#include "engine.hpp"
#include "common.hpp"
#include "quant.hpp"
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
    if (a.empty() || a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
template <typename T> T* device_alloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&p, n * sizeof(T)));
    return p;
}
// Isolate the new launch bound across every native format and T1..4. Each live CTA runs unchanged math;
// compare all scratch bytes as well as finite live outputs, including shared and sparse token routes.
void check_gu_launch_bounds() {
    constexpr int E = 2048, FF = 512, pairs = sq::kMaxU;
    std::mt19937 rng(261008);
    sq::ActQ x{device_alloc<int8_t>(sq::kMaxT * E), device_alloc<float>(sq::kMaxT * E / 32),
               device_alloc<float>(sq::kMaxT * E / 32)};
    std::vector<int8_t> q(sq::kMaxT * E);
    std::vector<float> d(q.size() / 32), sums(d.size());
    for (auto& v : q) v = (int8_t)((int)(rng() % 255) - 127);
    for (size_t b = 0; b < d.size(); ++b) {
        d[b] = (1.0f + rng() % 31) / 4096.0f;
        int sum = 0;
        for (int i = 0; i < 32; ++i) sum += q[b * 32 + i];
        sums[b] = sum * d[b];
    }
    CUDA_CHECK(cudaMemcpy(x.q, q.data(), q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x.d, d.data(), d.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x.s, sums.data(), sums.size() * sizeof(float), cudaMemcpyHostToDevice));
    auto* device_hits = device_alloc<sq::HitList>(1);
    float* output[2] = {device_alloc<float>(pairs * 2 * FF), device_alloc<float>(pairs * 2 * FF)};
    std::vector<float> result[2] = {std::vector<float>(pairs * 2 * FF), std::vector<float>(pairs * 2 * FF)};
    int cases = 0;
    for (uint32_t type : {sq::T_IQ2_S, sq::T_IQ3_S, sq::T_IQ4_XS, sq::T_Q2_K, sq::T_Q3_K}) {
        const size_t block_bytes = (size_t)sq::row_bytes(type, 256);
        std::vector<uint8_t> weights((size_t)sq::row_bytes(type, E) * 2 * FF);
        for (auto& b : weights) b = (uint8_t)rng();
        const int offset = type == sq::T_Q2_K ? 80 : type == sq::T_Q3_K ? 108 : 0;
        for (size_t b = 0; b < weights.size(); b += block_bytes) {
            const uint16_t scale = sq::f32_to_fp16((1.0f + rng() % 7) / 4096.0f);
            std::memcpy(weights.data() + b + offset, &scale, sizeof(scale));
            if (type == sq::T_Q2_K) std::memcpy(weights.data() + b + 82, &scale, sizeof(scale));
        }
        auto* device_weights = device_alloc<uint8_t>(weights.size());
        CUDA_CHECK(cudaMemcpy(device_weights, weights.data(), weights.size(), cudaMemcpyHostToDevice));
        for (int T : {1, 2, 3, 4}) for (int n : {1, 3, sq::kMaxK * T})
        for (bool reuse : {false, true}) for (bool transpose : {false, true}) {
            sq::HitList hits{};
            hits.n = n;
            std::memset(hits.pair, -1, sizeof(hits.pair));
            for (int j = 0; j < n; ++j) {
                hits.ptr[j] = device_weights;
                for (int t = 0; t < T; ++t) if (n == 1 || t == j % T) {
                    hits.pair[j][t] = (int8_t)hits.n_pair++;
                    hits.w[j][t] = T == 1 && j == 0 ? 0.0f : 0.125f; // T1 zero-weight still owns a pair
                }
            }
            SQ_CHECK(hits.n_pair <= sq::kMaxK * T, "invalid bounded-GU fixture");
            CUDA_CHECK(cudaMemcpy(device_hits, &hits, sizeof(hits), cudaMemcpyHostToDevice));
            for (int variant = 0; variant < 2; ++variant) {
                CUDA_CHECK(cudaMemset(output[variant], 0, result[variant].size() * sizeof(float)));
                sq::k_test_moe_gu(type, x, device_hits, output[variant], T, reuse, nullptr, transpose, variant ? n : 0);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(result[variant].data(), output[variant], result[variant].size() * sizeof(float), cudaMemcpyDeviceToHost));
            }
            SQ_CHECK(exact(result[0], result[1]), "bounded GU differs: %s T=%d n=%d reuse=%d transpose=%d",
                     sq::type_name(type), T, n, reuse, transpose);
            ++cases;
        }
        CUDA_CHECK(cudaFree(device_weights));
    }
    CUDA_CHECK(cudaFree(x.q)); CUDA_CHECK(cudaFree(x.d)); CUDA_CHECK(cudaFree(x.s));
    CUDA_CHECK(cudaFree(device_hits)); CUDA_CHECK(cudaFree(output[0])); CUDA_CHECK(cudaFree(output[1]));
    std::printf("bounded-vs-original GU grid: %d native-format/T1..4/shared/sparse/layout/reuse cases PASS\n", cases);
}
struct State {
    std::vector<float> logits;
    std::vector<uint8_t> prefix;
    std::vector<int> history, generated;
    uint64_t resident = 0, missing = 0, pairs = 0;
    std::mt19937_64 rng;
};
State execute(sq::Engine& engine, int batch, bool serialized) {
    env("STRATA_PROMPT_BATCH", std::to_string(batch).c_str());
    engine.debug_set_miss_stream_serialized(serialized);
    engine.reset();
    const int prompt[] = {198, 1000, 198, 1222, 198, 1333, 1444}; // odd tail exercises T3 with batch4
    const int window[] = {1555, 1666, 1777, 1888};
    const int continuation[] = {198, 1999, 2000};
    State result;
    auto append = [&](const std::vector<float>& values) {
        result.logits.insert(result.logits.end(), values.begin(), values.end());
    };
    uint64_t hits0, misses0, hits1, misses1;
    engine.expert_counts(hits0, misses0);
    const auto pairs0 = engine.stats.streamed_pairs;
    const auto experts0 = engine.stats.streamed_experts, bytes0 = engine.stats.streamed_bytes;
    const auto zero_cpu0 = engine.stats.stream_zero_cpu_layers;
    engine.stats.stream_max_unique = engine.stats.stream_max_pairs = 0; // per-execution coverage, not a model-state change
    engine.feed(prompt, 7);
    append(engine.logits_host());
    // Normalize pending-pair length before comparing prompt batches, while actually running MTP experts.
    append(engine.debug_mtp_catch_up(window[0], true));
    append(engine.debug_verify(window, 4, 2)); // keep2, discard2; recurrent/KV rollback must remain exact
    append(engine.logits_host());
    append(engine.debug_mtp_catch_up(continuation[0], true));
    engine.feed(continuation, 3);
    append(engine.logits_host());
    sq::SamplingParams sampling;
    sampling.temperature = 0.8f;
    sampling.top_k = 1;
    sampling.top_p = 1;
    sampling.min_p = 0;
    sampling.repetition_penalty = 1.05f;
    engine.begin_request(sampling);
    result.rng.seed(261008);
    const int next = engine.sample(sampling, result.rng);
    const std::vector<int> stops{-1};
    engine.spec_step(next, sampling, result.rng, result.generated, &stops, 4);
    append(engine.logits_host());
    result.history = engine.tokens();
    result.prefix = engine.debug_prefix_state();
    engine.expert_counts(hits1, misses1);
    result.resident = hits1 - hits0;
    result.missing = misses1 - misses0;
    result.pairs = engine.stats.streamed_pairs - pairs0;
    SQ_CHECK(engine.stats.streamed_experts > experts0 && result.pairs > 0 && engine.stats.streamed_bytes > bytes0,
             "transient path was bypassed");
    SQ_CHECK(result.pairs <= result.missing && (result.resident + result.missing) % engine.cfg().n_expert_used == 0,
             "streamed routes escaped original cache-miss/all-top8 accounting");
    if (engine.cache_slots() == 0)
        SQ_CHECK(result.resident == 0 && result.missing > 0, "zero-cache streamed routes were incorrectly counted as hits");
    SQ_CHECK(engine.stats.stream_max_unique <= sq::kMaxU && engine.stats.stream_max_pairs <= sq::kMaxU,
             "transient staging/pair bounds exceeded");
    if (engine.cache_slots() == 0 && engine.debug_miss_stream_eligible()[0].size() == (size_t)engine.cfg().n_expert)
        SQ_CHECK(engine.stats.stream_max_pairs == sq::kMaxU && engine.stats.stream_zero_cpu_layers > zero_cpu0,
                 "all-eligible fixture did not exercise all32 routed pairs and zero CPU remainder");
    SQ_CHECK(engine.debug_miss_stream_payloads_match(), "transient DMA payload changed original weights");
    std::printf("batch%d serialized%d streamed_experts=%llu pairs=%llu bytes=%llu resident=%llu missing=%llu max_unique=%llu max_pairs=%llu zero_cpu=%llu payload=PASS\n",
                batch, serialized, (unsigned long long)(engine.stats.streamed_experts - experts0),
                (unsigned long long)result.pairs, (unsigned long long)(engine.stats.streamed_bytes - bytes0),
                (unsigned long long)result.resident, (unsigned long long)result.missing,
                (unsigned long long)engine.stats.stream_max_unique, (unsigned long long)engine.stats.stream_max_pairs,
                (unsigned long long)(engine.stats.stream_zero_cpu_layers - zero_cpu0));
    return result;
}
void compare(const State& reference, const State& actual) {
    SQ_CHECK(exact(reference.logits, actual.logits), "full-vocabulary target/MTP/verification/continuation logits differ");
    SQ_CHECK(reference.prefix == actual.prefix, "canonical recurrent/KV/pending state differs");
    SQ_CHECK(reference.history == actual.history && reference.generated == actual.generated && reference.rng == actual.rng,
             "history/generated tokens/RNG differs");
    SQ_CHECK(reference.resident == actual.resident && reference.missing == actual.missing && reference.pairs == actual.pairs,
             "fixed-placement route counters differ");
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) sq::die("usage: test_miss_stream model.gguf [eligible_limit=2] [threads=6] [cache_mib=0]");
    const int eligible_limit = argc > 2 ? std::atoi(argv[2]) : 2;
    SQ_CHECK(eligible_limit >= 1 && eligible_limit <= 256, "eligible_limit must be1..256");
    env("STRATA_MISS_STREAM_EXPERTS", std::to_string(eligible_limit).c_str());
    env("STRATA_EVENT_MOE", "1");
    env("STRATA_SYNC_MOE", "0");
    env("STRATA_EVENT_MOE_GRAPH", "0");
    env("STRATA_EVENT_GRAPH_BLOCKING", "0");
    env("STRATA_PROFILE_PIPELINE", "0");
    env("STRATA_TRACE_STEPS", "0");
    env("STRATA_CACHE_REBALANCE_EVERY", "0");
    env("STRATA_KV_INITIAL_TOKENS", "0");
    env("STRATA_PROMPT_BATCH", "1");
    // Default covers all pending MTP rows. A separate parent run may set LAST_ROW=1 to cover the optimized
    // aliases/KV-only path; comparisons remain within that same mode, never across different counter semantics.
    if (!std::getenv("STRATA_MTP_LAST_ROW")) env("STRATA_MTP_LAST_ROW", "0");
    env("STRATA_NATIVE_EXPERTS", "1");
    sq::EngineOptions options;
    options.model_path = argv[1];
    options.ctx = 256;
    options.cache_mb = argc > 4 ? std::atoi(argv[4]) : 0;
    options.cpu_threads = argc > 3 ? std::atoi(argv[3]) : 6;
    options.vram_reserve_mb = 128;
    options.prefill_chunk = 0;
    options.mtp_draft = 3;
    options.draft_p = 0;
    options.ckpt_slots = 0;
    options.use_graph = false;
    options.adapt_every = 0;
    check_gu_launch_bounds(); // no model/cache allocation overlaps the small isolated kernel fixture
    {
        sq::Engine engine;
        std::string error;
        SQ_CHECK(engine.init(options, error), "%s", error.c_str());
        SQ_CHECK(engine.mtp(), "test requires MTP");
        std::vector<std::vector<int>> eligible(engine.cfg().n_moe_layers());
        engine.debug_set_miss_stream_eligible(eligible);
        const int probe = 198;
        engine.feed(&probe, 1);
        const auto first_ids = engine.debug_cpu_miss_ids(0);
        engine.debug_mtp_catch_up(1000, true);
        const auto mtp_ids = engine.debug_cpu_miss_ids(engine.cfg().n_layer);
        SQ_CHECK(first_ids.size() >= (size_t)std::min(eligible_limit, sq::kMaxK), "fixture needs enough layer0 misses; use cache_mib=0");
        auto fixed_ids = [&](const std::vector<int>& routed, int requested) {
            std::vector<int> ids(routed.begin(), routed.begin() + std::min(requested, (int)routed.size()));
            if (requested > sq::kMaxK) {
                for (int id = 0; (int)ids.size() < requested && id < engine.cfg().n_expert; ++id)
                    if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
            }
            std::sort(ids.begin(), ids.end());
            return ids;
        };
        std::vector<int> sizes{1};
        if (eligible_limit >= 2) sizes.push_back(2);
        if (eligible_limit >= 8) sizes.push_back(8);
        if (eligible_limit > 2 && eligible_limit != 8) sizes.push_back(eligible_limit);
        for (int selected : sizes) {
            eligible[0] = fixed_ids(first_ids, selected);
            eligible[engine.cfg().n_layer] = fixed_ids(mtp_ids, selected);
            engine.debug_set_miss_stream_eligible(eligible); // fixed for every async/serialized/T1..4 case below
            State reference = execute(engine, 1, true);
            for (int batch : {1, 2, 4}) {
                State actual = execute(engine, batch, false); // runs before payload readback; no sync can hide missing dependency
                compare(reference, actual);
                if (batch > 1) compare(actual, execute(engine, batch, true));
                SQ_CHECK(engine.debug_miss_stream_eligible() == eligible, "eligibility changed with token grouping");
                std::printf("fixed_eligible%d prompt_batch%d exact_DMA/full_logits/state/rollback/history/RNG/counters=PASS\n",
                            selected, batch);
            }
        }
        // Scope exit drains both CUDA streams before freeing staging and immutable pinned model sources.
    }
    std::puts("FIXED TRANSIENT MISS STREAM / T1..4 / SERIALIZED ORIGINAL-GRID PARITY / BOUNDED DMA / SHUTDOWN PASS");
    return 0;
}
