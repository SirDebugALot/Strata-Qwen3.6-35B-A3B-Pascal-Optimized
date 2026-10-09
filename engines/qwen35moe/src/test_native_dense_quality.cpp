// Full-model teacher-forced DQ8 vs original packed dense Q6 probe. Python owns locking/timeouts/memory guard.
// Usage: test_native_dense_quality model.gguf profile.bin manifest.tsv summary.json native_q6(0|1) [threads=6]
// Manifest: safe_case_id<TAB>prompt_count<TAB>ids_file<TAB>logits_file, one case per line.
// IDs contain the complete prompt then exactly eight fixed continuation tokens. Each output is
// nine consecutive full-vocabulary float32 rows: after prompt, then after each forced token.
// No sampling, generated continuations, adapting cache, prefix reuse, MTP, or lazy KV allocation.
#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}
struct Case { std::string id, output; int prompt; std::vector<int> ids; };
int integer(const std::string& text) {
    size_t n = 0;
    const int value = std::stoi(text, &n);
    if (n != text.size()) throw std::runtime_error("non-integer manifest value");
    return value;
}
std::vector<Case> manifest(const char* path) {
    std::ifstream source(path);
    if (!source) throw std::runtime_error("cannot open manifest");
    std::vector<Case> cases;
    std::string line;
    while (std::getline(source, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream fields(line);
        std::string id, count, input, output, extra;
        if (!std::getline(fields, id, '\t') || !std::getline(fields, count, '\t') ||
            !std::getline(fields, input, '\t') || !std::getline(fields, output, '\t') ||
            std::getline(fields, extra, '\t')) throw std::runtime_error("invalid manifest columns");
        if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
            throw std::runtime_error("invalid case id");
        for (const auto& old : cases) if (old.id == id || old.output == output)
            throw std::runtime_error("duplicate case id/output");
        std::ifstream ids_file(input);
        if (!ids_file) throw std::runtime_error("cannot open token file");
        std::string data((std::istreambuf_iterator<char>(ids_file)), std::istreambuf_iterator<char>());
        for (char& c : data) if (c == ',') c = ' ';
        std::istringstream tokens(data);
        Case c{id, output, integer(count), {}};
        int token;
        while (tokens >> token) c.ids.push_back(token);
        if (!tokens.eof() || c.prompt < 1 || c.prompt + 8 > 25000 || c.ids.size() != (size_t)c.prompt + 8)
            throw std::runtime_error("invalid prompt/continuation tokens");
        cases.push_back(std::move(c));
        if (cases.size() > 16) throw std::runtime_error("manifest exceeds bounded 16-case limit");
    }
    if (cases.empty()) throw std::runtime_error("empty manifest");
    return cases;
}

// Independent reconstruction of setup_cache's fixed-budget ordering. There is no adaptation or KV growth.
// The engine can only remove chosen layers on allocation failure: matching the exact full-budget slot/byte
// totals rules out those removals. Save every expected resident ID so the two modes can also compare placement.
std::vector<std::vector<int>> placement(const sq::Engine& engine, const sq::EngineOptions& opt) {
    const auto& c = engine.cfg();
    const int NL = c.n_moe_layers(), NE = c.n_expert;
    uint32_t header[3]{};
    std::ifstream f(opt.profile_path, std::ios::binary);
    if (!f.read((char*)header, sizeof(header)) || header[0] != 0x50455153 ||
        header[1] < (uint32_t)c.n_layer || header[1] > (uint32_t)c.n_layer + 1 || header[2] != (uint32_t)NE)
        throw std::runtime_error("invalid frozen profile header");
    std::vector<float> counts((size_t)header[1] * NE);
    if (!f.read((char*)counts.data(), counts.size() * sizeof(float))) throw std::runtime_error("short profile");
    std::vector<double> frequency((size_t)NL * NE, 0);
    const double live = (double)c.n_expert_used * std::max(opt.adapt_every, 1) / (1.0 - opt.adapt_decay);
    for (int layer = 0; layer < std::min(NL, (int)header[1]); ++layer) {
        double sum = 0;
        for (int e = 0; e < NE; ++e) {
            const float value = counts[(size_t)layer * NE + e];
            if (!std::isfinite(value) || value < 0) throw std::runtime_error("invalid profile count");
            sum += value;
        }
        const double scale = sum > 0 ? opt.profile_weight * live / sum : 0;
        for (int e = 0; e < NE; ++e) frequency[(size_t)layer * NE + e] = counts[(size_t)layer * NE + e] * scale;
    }
    std::vector<int> order((size_t)NL * NE);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return frequency[a] != frequency[b] ? frequency[a] > frequency[b] : a % NE < b % NE;
    });
    std::vector<std::vector<int>> chosen(NL);
    int64_t used = 0, slots = 0;
    const int64_t budget = opt.cache_mb * 1048576ll;
    for (int index : order) {
        const int layer = index / NE;
        const int64_t bytes = engine.model().layers[layer].blob_bytes;
        if (used + bytes > budget) continue;
        used += bytes; ++slots;
        chosen[layer].push_back(index % NE);
    }
    if (engine.cache_slots() != slots || std::llround(engine.cache_gib() * 1073741824.0) != used)
        throw std::runtime_error("cache placement was reduced below deterministic fixed-budget plan");
    return chosen;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 6 || argc > 7) {
            std::cerr << "usage: test_native_dense_quality model.gguf profile.bin manifest.tsv summary.json native_q6(0|1) [threads=6]\n";
            return 2;
        }
        const auto cases = manifest(argv[3]);
        const int native_q6 = integer(argv[5]);
        const int cache = 512;
        const int threads = argc > 6 ? integer(argv[6]) : 6;
        if ((native_q6 != 0 && native_q6 != 1) || threads < 1 || threads > 12)
            throw std::runtime_error("native_q6 must be 0/1; threads must be 1..12");
        if (!std::ifstream(argv[2])) throw std::runtime_error("missing frozen expert profile");
        env("STRATA_NATIVE_EXPERTS", "1");
        env("STRATA_EXPERT_CACHE_FILE", "");
        env("STRATA_CPU_LM_HEAD", "0");
        env("STRATA_NATIVE_DENSE_Q6", native_q6 ? "1" : "0");
        env("STRATA_KV_INT8", "0");
        env("STRATA_CPU_ISA", "avx2");
        env("STRATA_EVENT_MOE", "1");
        env("STRATA_SYNC_MOE", "0");
        env("STRATA_ASYNC_CACHE", "0");
        env("STRATA_CACHE_REBALANCE_EVERY", "0");
        env("STRATA_KV_INITIAL_TOKENS", "0");
        env("STRATA_PROMPT_BATCH", "4");
        env("STRATA_MTP_DRAFT_SUBSET", "0");
        env("STRATA_PROFILE_PIPELINE", "0");
        env("STRATA_TRACE_STEPS", "0");
        env("STRATA_CACHE_INIT_RESERVE_MIB", "0");
        env("STRATA_CACHE_BUDGET_EXTRA_MIB", "0");
        sq::EngineOptions opt;
        opt.model_path = argv[1]; opt.profile_path = argv[2];
        opt.ctx = 25000; opt.cpu_threads = threads; opt.cache_mb = cache;
        opt.vram_reserve_mb = 128; opt.prefill_chunk = 0; opt.mtp_draft = 0;
        opt.ckpt_slots = 0; opt.use_graph = false; opt.adapt_every = 0;
        opt.adapt_decay = 0.85; opt.profile_weight = 0.25;
        sq::Engine engine;
        std::string error;
        if (!engine.init(opt, error)) throw std::runtime_error(error);
        if (engine.model().cpu_lm_head || engine.model().native_dense_q6 != (native_q6 != 0) ||
            engine.mtp() || std::string(engine.kv_type()) != "fp16" || engine.kv_capacity() != engine.ctx())
            throw std::runtime_error("engine settings differ from controlled dense-Q6 plan");
        const auto chosen = placement(engine, opt);
        const int64_t cache_bytes = (int64_t)std::llround(engine.cache_gib() * 1073741824.0);
        std::ofstream summary(argv[4], std::ios::trunc);
        if (!summary) throw std::runtime_error("cannot open summary output");
        summary << "{\"native_dense_q6\":" << (native_q6 ? "true" : "false")
                << ",\"gpu_head\":true,\"mtp\":false,\"kv_type\":\"" << engine.kv_type() << "\",\"context\":" << engine.ctx()
                << ",\"kv_capacity\":" << engine.kv_capacity() << ",\"kv_bytes\":" << engine.kv_bytes()
                << ",\"cache_slots\":" << engine.cache_slots() << ",\"cache_bytes\":" << cache_bytes
                << ",\"vocab\":" << engine.cfg().n_vocab << ",\"resident_ids_by_layer\":[";
        for (size_t layer = 0; layer < chosen.size(); ++layer) {
            if (layer) summary << ',';
            summary << '[';
            for (size_t slot = 0; slot < chosen[layer].size(); ++slot) {
                if (slot) summary << ',';
                summary << chosen[layer][slot];
            }
            summary << ']';
        }
        summary << "],\"cases\":[";
        bool all_finite = true;
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto& c = cases[i];
            for (int token : c.ids) if (token < 0 || token >= engine.cfg().n_vocab)
                throw std::runtime_error("token outside vocabulary");
            engine.reset();
            engine.feed(c.ids.data(), c.prompt);
            std::ofstream dump(c.output, std::ios::binary | std::ios::trunc);
            if (!dump) throw std::runtime_error("cannot open logits output");
            bool finite = true;
            for (int offset = 0; offset <= 8; ++offset) {
                if (offset) engine.feed(c.ids.data() + c.prompt + offset - 1, 1);
                const auto logits = engine.logits_host();
                if (logits.size() != (size_t)engine.cfg().n_vocab || engine.n_past() != c.prompt + offset ||
                    engine.tokens().size() != (size_t)c.prompt + offset ||
                    !std::equal(engine.tokens().begin(), engine.tokens().end(), c.ids.begin()))
                    throw std::runtime_error("logits/history/position mismatch");
                for (float value : logits) finite = finite && std::isfinite(value);
                dump.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
                if (!dump) throw std::runtime_error("failed writing logits");
            }
            dump.close();
            if (!dump) throw std::runtime_error("failed closing logits");
            all_finite = all_finite && finite;
            if (i) summary << ',';
            summary << "{\"id\":\"" << c.id << "\",\"prompt_tokens\":" << c.prompt
                    << ",\"continuation_tokens\":8,\"rows\":9,\"all_finite\":" << (finite ? "true" : "false") << '}';
            std::cout << "case=" << c.id << " prompt=" << c.prompt << " rows=9 finite=" << finite << std::endl;
        }
        summary << "]}\n";
        summary.close();
        if (!summary) throw std::runtime_error("failed writing summary");
        return all_finite ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Native dense quality probe: " << e.what() << '\n';
        return 2;
    }
}
