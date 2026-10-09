// Opt-in microbatch feed parity, including the LAST logits row and recurrent/KV state carried into later tokens.
// Link with sq_core. Usage: test_prompt_batch model.gguf [prompt_tokens=257] [cache_mib=128] [threads=6] [mtp=0]
// This test starts one model. Run only when the GPU is free. STRATA_EXPERT_NATIVE/CPU_LM_HEAD are caller-selected.
// Synthetic tokens test state/indexing, not language quality; actual English/Korean/code prompts remain required.
#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {
void set_env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

bool compare(const std::vector<float>& ref, const std::vector<float>& got, int batch, int continuation) {
    if (ref.size() != got.size() || ref.empty()) return false;
    double mr = -INFINITY, mg = -INFINITY, sr = 0, sg = 0;
    size_t ar = 0, ag = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) return false;
        mr = std::max(mr, (double)ref[i]);
        mg = std::max(mg, (double)got[i]);
        if (ref[i] > ref[ar]) ar = i;
        if (got[i] > got[ag]) ag = i;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        sr += std::exp(ref[i] - mr);
        sg += std::exp(got[i] - mg);
    }
    double kl = 0, max_logit = 0, max_active_logp = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double lr = ref[i] - mr - std::log(sr), lg = got[i] - mg - std::log(sg);
        kl += std::exp(lr) * (lr - lg);
        max_logit = std::max(max_logit, std::abs((double)ref[i] - got[i]));
        if (lr > -10.0) max_active_logp = std::max(max_active_logp, std::abs(lr - lg));
    }
    const bool ok = std::isfinite(kl) && kl <= 1e-4 && max_active_logp <= 0.10 && ar == ag;
    std::printf("batch=%d continuation=%d KL=%.9g max_dlogit=%.9g active_dlogp=%.9g top1=%zu/%zu %s\n",
                batch, continuation, kl, max_logit, max_active_logp, ar, ag, ok ? "PASS" : "FAIL");
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_prompt_batch model.gguf [prompt_tokens=257] [cache_mib=128] [threads=6] [mtp=0]\n");
        return 2;
    }
    const int n = argc > 2 ? std::atoi(argv[2]) : 257;
    const int mtp = argc > 5 ? std::atoi(argv[5]) : 0;
    if (n < 4 || n > 4096 || mtp < 0 || mtp >= sq::kMaxT) {
        std::fprintf(stderr, "prompt_tokens must be 4..4096 and mtp 0..3\n");
        return 2;
    }
    set_env("STRATA_EVENT_MOE", "1");
    set_env("STRATA_SYNC_MOE", "0");
    set_env("STRATA_PROMPT_BATCH", "1");
    set_env("STRATA_ASYNC_CACHE", "0");
    set_env("STRATA_CACHE_REBALANCE_EVERY", "0");
    set_env("STRATA_KV_INITIAL_TOKENS", "0"); // KV growth trims cache slots; isolate batching with frozen placement
    sq::EngineOptions opt;
    opt.model_path = argv[1];
    opt.ctx = n + 256;
    opt.cache_mb = argc > 3 ? std::atoi(argv[3]) : 128;
    opt.cpu_threads = argc > 4 ? std::atoi(argv[4]) : 6;
    opt.vram_reserve_mb = 128;
    opt.prefill_chunk = 0;
    opt.mtp_draft = mtp;
    opt.ckpt_slots = 0;
    opt.use_graph = false;
    opt.adapt_every = 0;  // Frozen placement isolates arithmetic/state from adaptation.
    sq::Engine eng;
    std::string err;
    if (!eng.init(opt, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (mtp > 0 && !eng.mtp()) {
        std::fprintf(stderr, "requested MTP block was not loaded\n");
        return 1;
    }
    const char* last_row_env = std::getenv("STRATA_MTP_LAST_ROW");
    const bool last_row = eng.mtp() && last_row_env && std::atoi(last_row_env) == 1;
    std::mt19937 rng(61028);
    std::vector<int> tokens(n + 8 + sq::kMaxT);
    const int range = std::min(50000, eng.cfg().n_vocab - 1000);
    if (range <= 0) return 2;
    for (int& token : tokens) token = 1000 + (int)(rng() % range);
    std::vector<std::vector<float>> reference;
    std::vector<float> verify_reference, rollback_reference;
    bool ok = true;
    for (int batch : {1, 2, 4}) {
        set_env("STRATA_PROMPT_BATCH", std::to_string(batch).c_str());
        eng.reset();
        const int effective_batch = eng.mtp() ? std::min(batch, opt.mtp_draft + 1) : batch;
        std::printf("prompt batch requested=%d effective=%d MTP=%d\n", batch, effective_batch, eng.mtp_draft());
        uint64_t h0 = 0, m0 = 0, h1 = 0, m1 = 0;
        eng.expert_counts(h0, m0);
        const auto steps0 = eng.stats.decode_steps;
        const auto kv_only0 = eng.debug_mtp_kv_only_steps(), last_row0 = eng.debug_mtp_last_row_steps();
        if (batch == 1) {
            for (int i = 0; i < n; ++i) eng.feed(tokens.data() + i, 1);
        } else {
            eng.feed(tokens.data(), n);  // Whole prompt, including a non-multiple tail.
        }
        for (int continuation = 0; continuation <= 8; ++continuation) {
            if (continuation) eng.feed(tokens.data() + n + continuation - 1, 1);
            const auto logits = eng.logits_host();
            if (batch == 1) reference.push_back(logits);
            else ok = compare(reference[continuation], logits, batch, continuation) && ok;
            const int expected = n + continuation;
            if (eng.n_past() != expected || eng.tokens().size() != (size_t)expected ||
                !std::equal(eng.tokens().begin(), eng.tokens().end(), tokens.begin())) {
                std::fprintf(stderr, "batch %d: prompt history/position mismatch at %d\n", batch, expected);
                ok = false;
            }
        }
        eng.expert_counts(h1, m1);
        // These calls only feed known tokens. With last-row mode their MTP catch-ups update KV without
        // routing any experts; otherwise every completed pair retains the original one-layer MoE count.
        const uint64_t mtp_pairs = eng.mtp() && !last_row ? n + 8 - 1 : 0;
        const uint64_t expected_lookups = ((uint64_t)(n + 8) * eng.cfg().n_layer + mtp_pairs) * eng.cfg().n_expert_used;
        const uint64_t lookups = (h1 - h0) + (m1 - m0);
        // One catch-up precedes each prompt chunk except the first, then each of eight serial continuation
        // calls catches up the previous chunk/token. The last token's hidden pair remains pending.
        const uint64_t expected_kv_only = last_row ? (uint64_t)((n + effective_batch - 1) / effective_batch - 1 + 8) : 0;
        const uint64_t kv_only = eng.debug_mtp_kv_only_steps() - kv_only0;
        const uint64_t last_rows = eng.debug_mtp_last_row_steps() - last_row0;
        if (lookups != expected_lookups || eng.stats.decode_steps - steps0 != (uint64_t)(n + 8) ||
            kv_only != expected_kv_only || last_rows != 0) {
            std::fprintf(stderr, "batch %d: counters lookups=%llu expected=%llu steps=%llu KV-only=%llu/%llu draft-tails=%llu\n", batch,
                         (unsigned long long)lookups, (unsigned long long)expected_lookups,
                         (unsigned long long)(eng.stats.decode_steps - steps0),
                         (unsigned long long)kv_only, (unsigned long long)expected_kv_only, (unsigned long long)last_rows);
            ok = false;
        }
        std::printf("batch=%d routed_entries=%llu expected=%llu KV-only=%llu/%llu last-row-mode=%d\n", batch,
                    (unsigned long long)lookups, (unsigned long long)expected_lookups,
                    (unsigned long long)kv_only, (unsigned long long)expected_kv_only, last_row);
        if (eng.mtp()) {
            // Flush the final pending MTP pair, verify a full allocated window, then keep only its first token.
            // The following feed catches up that kept pair and tests state after rollback as well as the prompt.
            const int T = opt.mtp_draft + 1;
            const auto verification = eng.debug_verify(tokens.data() + n + 8, T, 1);
            if (batch == 1) verify_reference = verification;
            else {
                const size_t vocab = eng.cfg().n_vocab;
                for (int t = 0; t < T; ++t) {
                    const std::vector<float> expected(verify_reference.begin() + t * vocab,
                                                      verify_reference.begin() + (t + 1) * vocab);
                    const std::vector<float> actual(verification.begin() + t * vocab,
                                                    verification.begin() + (t + 1) * vocab);
                    ok = compare(expected, actual, batch, 100 + t) && ok;
                }
            }
            eng.feed(tokens.data() + n + 9, 1);
            const auto after_rollback = eng.logits_host();
            if (batch == 1) rollback_reference = after_rollback;
            else ok = compare(rollback_reference, after_rollback, batch, 200) && ok;
            if (eng.n_past() != n + 10 || eng.tokens().size() != (size_t)(n + 10) ||
                !std::equal(eng.tokens().begin(), eng.tokens().end(), tokens.begin())) {
                std::fprintf(stderr, "batch %d: MTP verification rollback/history mismatch\n", batch);
                ok = false;
            }
        }
    }
    std::puts(ok ? "PROMPT BATCH PARITY PASS" : "PROMPT BATCH PARITY FAIL");
    return ok ? 0 : 1;
}
