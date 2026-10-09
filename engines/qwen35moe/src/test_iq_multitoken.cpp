// CPU-only native IQ row reuse and host-dispatched MoE batch parity.
// test_iq_multitoken <model.gguf> [threads]
#include "common.hpp"
#include "cpu_moe.hpp"
#include "gguf.hpp"
#include "native_iq/iq_avx2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace sq;
namespace iq = strata::kernels::cpu;

static std::string tensor_name(int layer, const char* name) {
    return "blk." + std::to_string(layer) + "." + name;
}

static double scaled_error(float actual, float expected) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) return 1e30;
    return std::fabs((double)actual - expected) / (1.0 + std::fabs((double)expected));
}

static void set_gather_env(const char* value) {
#ifdef _WIN32
    _putenv_s("STRATA_IQ256_GATHER", value ? value : "");
#else
    if (value) setenv("STRATA_IQ256_GATHER", value, 1);
    else unsetenv("STRATA_IQ256_GATHER");
#endif
}

int main(int argc, char** argv) {
    if (argc < 2) die("usage: test_iq_multitoken <model.gguf> [threads]");
    const int threads = argc > 2 ? std::atoi(argv[2]) : 6;
    SQ_CHECK(threads > 0, "threads must be positive");
    SQ_CHECK(cpu_force_isa("avx2"), "native IQ test requires AVX2/FMA/F16C");
    const char* gather_env = std::getenv("STRATA_IQ256_GATHER");
    const bool had_gather_env = gather_env != nullptr;
    const std::string original_gather_env = gather_env ? gather_env : "";
    const int cached_variant = iq::iq256_variant();
    set_gather_env((cached_variant & iq::kIq256Gather) ? "0" : "1");
    SQ_CHECK(iq::iq256_variant() == cached_variant, "IQ gather setting was not cached per process");
    set_gather_env(had_gather_env ? original_gather_env.c_str() : nullptr);
    GgufFile model;
    std::string error;
    if (!model.open(argv[1], error)) die("%s", error.c_str());
    std::mt19937 rng(7026);
    std::normal_distribution<float> normal(0.f, 1.f);
    int failures = 0, row_cases = 0, layer_cases = 0;

    // Each format/row-width present in the model, all window sizes, and both AVX2 lookup variants.
    std::set<std::pair<uint32_t, int>> row_seen;
    for (const auto& tensor : model.tensors()) {
        if (!iq::iq256_supported(tensor.type) || tensor.ne.empty() || tensor.ne[0] % 256 != 0) continue;
        const int cols = (int)tensor.ne[0];
        if (!row_seen.emplace(tensor.type, cols).second) continue;
        constexpr int rows = 16;
        if (tensor.n_elements() / cols < rows) continue;
        const int64_t rb = row_bytes(tensor.type, cols);
        std::vector<Q8K> activations((size_t)kMaxT * cols / 256);
        const void* act[kMaxT];
        float output[kMaxT][rows];
        float* out[kMaxT];
        std::vector<float> x(cols);
        for (int t = 0; t < kMaxT; ++t) {
            for (float& value : x) value = normal(rng) * (0.3f + t);
            Q8K* q = activations.data() + (size_t)t * cols / 256;
            quantize_q8k(x.data(), q, cols);
            act[t] = q;
            out[t] = output[t];
        }
        for (int variant : {0, iq::kIq256Gather}) {
            for (int nt = 1; nt <= kMaxT; ++nt) {
                iq::iq256_rows_v(variant, tensor.type, tensor.data, (size_t)rb, cols, act, nt, out, 0, rows);
                if (variant == cached_variant) {
                    float explicit_output[kMaxT][rows];
                    std::memcpy(explicit_output, output, (size_t)nt * rows * sizeof(float));
                    iq::iq256_rows(tensor.type, tensor.data, (size_t)rb, cols, act, nt, out, 0, rows);
                    SQ_CHECK(std::memcmp(explicit_output, output, (size_t)nt * rows * sizeof(float)) == 0,
                             "cached automatic variant differs from explicit variant %d", variant);
                }
                double worst = 0;
                for (int t = 0; t < nt; ++t)
                    for (int r = 0; r < rows; ++r) {
                        const float expected = dot_row(tensor.type, tensor.data + r * rb,
                                                       (const Q8K*)act[t], cols / 256);
                        worst = std::max(worst, scaled_error(output[t][r], expected));
                    }
                const bool ok = worst < 2e-5;
                failures += !ok;
                ++row_cases;
                log("IQ rows %s cols %d T%d variant %d max scaled error %.3g %s",
                    type_name(tensor.type), cols, nt, variant, worst, ok ? "OK" : "FAIL");
            }
        }
    }

    // Exercise the real worker pool and both reused projections, including experts shared by all tokens,
    // partial overlap, and the full 32-distinct-expert mailbox.  Repeated T1 calls are the reference.
    std::set<std::pair<uint32_t, uint32_t>> layer_seen;
    for (int layer = 0;; ++layer) {
        const auto* gate = model.tensor(tensor_name(layer, "ffn_gate_exps.weight"));
        if (!gate) break;
        const auto& up = model.need(tensor_name(layer, "ffn_up_exps.weight"));
        const auto& down = model.need(tensor_name(layer, "ffn_down_exps.weight"));
        if (!layer_seen.emplace(gate->type, down.type).second) continue;
        SQ_CHECK(gate->ne[0] == 2048 && down.ne[0] == 512 && gate->type == up.type,
                 "unexpected Qwen expert layout in layer %d", layer);
        ExpertLayerDesc desc;
        desc.t_gu = gate->type;
        desc.t_down = down.type;
        desc.gu_bytes = row_bytes(gate->type, 2048) * 512;
        const int64_t down_bytes = row_bytes(down.type, 512) * 2048;
        desc.blob = 2 * desc.gu_bytes + down_bytes;
        SQ_CHECK(gate->bytes >= kMaxU * desc.gu_bytes && down.bytes >= kMaxU * down_bytes,
                 "model has too few experts for the 32-expert test");
        std::vector<uint8_t> blobs((size_t)kMaxU * desc.blob);
        for (int e = 0; e < kMaxU; ++e) {
            uint8_t* dst = blobs.data() + e * desc.blob;
            std::memcpy(dst, gate->data + e * desc.gu_bytes, (size_t)desc.gu_bytes);
            std::memcpy(dst + desc.gu_bytes, up.data + e * desc.gu_bytes, (size_t)desc.gu_bytes);
            std::memcpy(dst + 2 * desc.gu_bytes, down.data + e * down_bytes, (size_t)down_bytes);
        }
        desc.base = blobs.data();
        auto mailbox = std::make_unique<Mailbox>();
        auto result = std::make_unique<Result>();
        CpuMoe cpu;
        cpu.start(threads, 1, {desc}, mailbox.get(), result.get(), false);
        for (int nt = 1; nt <= kMaxT; ++nt) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                std::memset(mailbox.get(), 0, sizeof(Mailbox));
                mailbox->n_tok = nt;
                for (int t = 0; t < nt; ++t)
                    for (float& value : mailbox->x[t]) value = normal(rng);
                for (int e = 0; e < kMaxU; ++e) {
                    float weights[kMaxT] = {};
                    bool used = false;
                    for (int t = 0; t < nt; ++t) {
                        const int first = pattern == 0 ? 0 : pattern == 1 ? 4 * t : 8 * t;
                        if (e >= first && e < first + kMaxK) {
                            weights[t] = (1.f + (e - first)) / 36.f;
                            used = true;
                        }
                    }
                    if (!used) continue;
                    const int j = mailbox->n_miss++;
                    mailbox->ids[j] = e;
                    std::memcpy(mailbox->w[j], weights, sizeof(weights));
                }
                cpu.begin_layer(0);
                cpu.end_layer();
                double worst = 0;
                for (int t = 0; t < nt; ++t) {
                    int ids[kMaxK], n = 0;
                    float weights[kMaxK], reference[2048];
                    for (int j = 0; j < mailbox->n_miss; ++j) {
                        if (mailbox->w[j][t] == 0.f) continue;
                        ids[n] = mailbox->ids[j];
                        weights[n++] = mailbox->w[j][t];
                    }
                    cpu.compute_layer_sync(0, mailbox->x[t], ids, weights, n, reference);
                    for (int r = 0; r < 2048; ++r)
                        worst = std::max(worst, scaled_error(result->out[t][r], reference[r]));
                }
                const bool ok = worst < 2e-5;
                failures += !ok;
                ++layer_cases;
                log("CPU batch layer %d %s/%s T%d pattern %d distinct %d max scaled error %.3g %s",
                    layer, type_name(desc.t_gu), type_name(desc.t_down), nt, pattern, mailbox->n_miss,
                    worst, ok ? "OK" : "FAIL");
            }
        }
        cpu.stop();
    }
    SQ_CHECK(row_cases > 0 && layer_cases > 0, "no applicable IQ rows or MoE layers found");
    log("native IQ multi-token: %d row cases, %d worker cases, %d failures", row_cases, layer_cases, failures);
    return failures ? 1 : 0;
}
