// Full-model teacher-forced KV probe. Link sq_core; the Python driver owns locking/timeouts.
// Usage: test_kv_quality model.gguf profile.bin manifest.tsv summary.json [cache_mib=512] [threads=6]
// Manifest: safe_case_id<TAB>prompt_count<TAB>ids_file<TAB>logits_file, one case per line.
// IDs contain the complete prompt then exactly eight fixed continuation tokens. Each output is
// nine consecutive full-vocabulary float32 rows: after prompt, then after each forced token.
// No sampling, generated continuations, adapting cache, prefix reuse, MTP, or lazy KV allocation.
#include "engine.hpp"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
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
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 7) {
            std::cerr << "usage: test_kv_quality model.gguf profile.bin manifest.tsv summary.json [cache_mib=512] [threads=6]\n";
            return 2;
        }
        const auto cases = manifest(argv[3]);
        const int cache = argc > 5 ? integer(argv[5]) : 512;
        const int threads = argc > 6 ? integer(argv[6]) : 6;
        if ((cache != 512 && cache != 1024) || threads < 1 || threads > 12)
            throw std::runtime_error("cache must be 512/1024 MiB; threads must be 1..12");
        if (!std::ifstream(argv[2])) throw std::runtime_error("missing frozen expert profile");
        env("STRATA_NATIVE_EXPERTS", "1");
        env("STRATA_EXPERT_CACHE_FILE", "");
        env("STRATA_CPU_LM_HEAD", "1");
        env("STRATA_CPU_ISA", "avx2");
        env("STRATA_EVENT_MOE", "1");
        env("STRATA_SYNC_MOE", "0");
        env("STRATA_ASYNC_CACHE", "0");
        env("STRATA_CACHE_REBALANCE_EVERY", "0");
        env("STRATA_KV_INITIAL_TOKENS", "0");
        env("STRATA_PROMPT_BATCH", "4");
        sq::EngineOptions opt;
        opt.model_path = argv[1]; opt.profile_path = argv[2];
        opt.ctx = 25000; opt.cpu_threads = threads; opt.cache_mb = cache;
        opt.vram_reserve_mb = 128; opt.prefill_chunk = 0; opt.mtp_draft = 0;
        opt.ckpt_slots = 0; opt.use_graph = false; opt.adapt_every = 0;
        sq::Engine engine;
        std::string error;
        if (!engine.init(opt, error)) throw std::runtime_error(error);
        const int64_t cache_bytes = (int64_t)std::llround(engine.cache_gib() * 1073741824.0);
        std::ofstream summary(argv[4], std::ios::trunc);
        if (!summary) throw std::runtime_error("cannot open summary output");
        summary << "{\"kv_type\":\"" << engine.kv_type() << "\",\"context\":" << engine.ctx()
                << ",\"kv_capacity\":" << engine.kv_capacity() << ",\"kv_bytes\":" << engine.kv_bytes()
                << ",\"cache_slots\":" << engine.cache_slots() << ",\"cache_bytes\":" << cache_bytes
                << ",\"vocab\":" << engine.cfg().n_vocab << ",\"cases\":[";
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
        std::cerr << "KV quality probe: " << e.what() << '\n';
        return 2;
    }
}
