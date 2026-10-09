// test_cpu_kernels.cpp - parity of the expert dot products against scalar dequantization, a full expert-layer
// check, and CPU MoE throughput - for each instruction set this CPU runs (AVX-512 VNNI, AVX2).
//   test_cpu_kernels <model.gguf> [threads]
#include "common.hpp"
#include "cpu_moe.hpp"
#include "gguf.hpp"
#include "quant.hpp"

#include <cmath>
#include <random>
#include <string>
#include <vector>

using namespace sq;

static std::string blk(int il, const char* n) { return "blk." + std::to_string(il) + "." + n; }

int main(int argc, char** argv) {
    if (argc < 2) die("usage: test_cpu_kernels <model.gguf> [threads]");
    const int threads = argc > 2 ? std::atoi(argv[2]) : 8;
    GgufFile f;
    std::string err;
    if (!f.open(argv[1], err)) die("%s", err.c_str());
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);

    int fails = 0;
    for (const char* isa : {"avx512", "avx2"}) {
    if (!cpu_force_isa(isa)) {
        log("== %s: not supported by this CPU, skipped", isa);
        continue;
    }
    log("== %s kernels", cpu_isa());
    // ---------------------------------------------------------------- per-row parity
    struct Case { int layer; const char* name; int cols; };
    std::vector<Case> cases;
    for (int il = 0; il < 41; ++il) {
        if (!f.tensor(blk(il, "ffn_gate_exps.weight"))) break;
        const auto& g = f.need(blk(il, "ffn_gate_exps.weight"));
        const auto& d = f.need(blk(il, "ffn_down_exps.weight"));
        static bool seen_g[32] = {}, seen_d[32] = {};
        if (il == 0) std::fill(seen_g, seen_g + 32, false), std::fill(seen_d, seen_d + 32, false);
        if (!seen_g[g.type]) { seen_g[g.type] = true; cases.push_back({il, "ffn_gate_exps.weight", 2048}); }
        if (!seen_d[d.type]) { seen_d[d.type] = true; cases.push_back({il, "ffn_down_exps.weight", 512}); }
    }
    for (const Case& c : cases) {
        const auto& t = f.need(blk(c.layer, c.name));
        std::vector<float> x(c.cols);
        for (auto& v : x) v = nd(rng) * 3.0f;
        std::vector<Q8K> xq(c.cols / 256);
        quantize_q8k(x.data(), xq.data(), c.cols);
        const int64_t rb = row_bytes(t.type, c.cols);
        double max_rel = 0;
        for (int r = 0; r < 2000; ++r) {
            const int64_t row = (int64_t)(rng() % (t.n_elements() / c.cols));
            const uint8_t* w = t.data + row * rb;
            const float a = dot_row(t.type, w, xq.data(), c.cols / 256);
            const float b = dot_row_ref(t.type, w, xq.data(), c.cols / 256);
            std::vector<float> wf(c.cols);
            dequant_row(t.type, w, wf.data(), c.cols);
            double mag = 0;
            for (int k = 0; k < c.cols; ++k) mag += std::fabs(wf[k] * x[k]);
            max_rel = std::max(max_rel, std::fabs((double)a - b) / (mag + 1e-9));
        }
        const bool ok = max_rel < 1e-5;
        fails += !ok;
        log("dot %-5s layer %2d %-22s max |simd-ref|/sum|w*x| = %.3g  %s", type_name(t.type), c.layer, c.name, max_rel,
            ok ? "OK" : "FAIL");
    }

    // ---------------------------------------------------------------- full expert layer vs float reference
    const int n_layer = 40;
    std::vector<ExpertLayerDesc> descs(n_layer);
    std::vector<std::vector<uint8_t>> blobs(n_layer);
    const int test_experts = 16;   // only the first 16 experts are packed, to keep the test light
    for (int il = 0; il < n_layer; ++il) {
        const auto& g = f.need(blk(il, "ffn_gate_exps.weight"));
        const auto& u = f.need(blk(il, "ffn_up_exps.weight"));
        const auto& d = f.need(blk(il, "ffn_down_exps.weight"));
        ExpertLayerDesc& D = descs[il];
        D.t_gu = g.type;
        D.t_down = d.type;
        D.gu_bytes = row_bytes(g.type, 2048) * 512;
        const int64_t db = row_bytes(d.type, 512) * 2048;
        D.blob = 2 * D.gu_bytes + db;
        blobs[il].resize((size_t)(D.blob * test_experts));
        for (int e = 0; e < test_experts; ++e) {
            uint8_t* dst = blobs[il].data() + e * D.blob;
            std::memcpy(dst, g.data + e * D.gu_bytes, D.gu_bytes);
            std::memcpy(dst + D.gu_bytes, u.data + e * D.gu_bytes, D.gu_bytes);
            std::memcpy(dst + 2 * D.gu_bytes, d.data + e * db, db);
        }
        D.base = blobs[il].data();
    }
    CpuMoe moe;
    moe.start(threads, n_layer, descs, nullptr, nullptr, true);
    for (int il : {0, 3, 39}) {
        std::vector<float> x(2048);
        for (auto& v : x) v = nd(rng);
        const int ids[8] = {0, 3, 5, 7, 9, 11, 13, 15};
        const float wts[8] = {0.3f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f, 0.05f, 0.05f};
        std::vector<float> out(2048), ref(2048, 0.0f);
        moe.compute_layer_sync(il, x.data(), ids, wts, 8, out.data());
        const ExpertLayerDesc& D = descs[il];
        std::vector<float> wg(2048), wu(2048), wd(512), h(512);
        for (int j = 0; j < 8; ++j) {
            const uint8_t* b = D.base + ids[j] * D.blob;
            for (int r = 0; r < 512; ++r) {
                dequant_row(D.t_gu, b + r * row_bytes(D.t_gu, 2048), wg.data(), 2048);
                dequant_row(D.t_gu, b + D.gu_bytes + r * row_bytes(D.t_gu, 2048), wu.data(), 2048);
                double g = 0, u = 0;
                for (int k = 0; k < 2048; ++k) { g += wg[k] * x[k]; u += wu[k] * x[k]; }
                h[r] = (float)(g / (1 + std::exp(-g)) * u);
            }
            for (int r = 0; r < 2048; ++r) {
                dequant_row(D.t_down, b + 2 * D.gu_bytes + r * row_bytes(D.t_down, 512), wd.data(), 512);
                double s = 0;
                for (int k = 0; k < 512; ++k) s += wd[k] * h[k];
                ref[r] += wts[j] * (float)s;
            }
        }
        double num = 0, den = 0;
        for (int r = 0; r < 2048; ++r) { num += (out[r] - ref[r]) * (out[r] - ref[r]); den += ref[r] * ref[r]; }
        const double rel = std::sqrt(num / den);
        const bool ok = rel < 3e-2;   // int8 activations (x and h), same as ggml's CPU path
        fails += !ok;
        log("expert layer %2d (%s/%s): rel L2 error vs float reference = %.4f  %s", il, type_name(D.t_gu),
            type_name(D.t_down), rel, ok ? "OK" : "FAIL");
    }

    // ---------------------------------------------------------------- throughput (8 experts per layer)
    {
        std::vector<float> x(2048), out(2048);
        for (auto& v : x) v = nd(rng);
        const float wts[8] = {.125f, .125f, .125f, .125f, .125f, .125f, .125f, .125f};
        for (int n : {1, 3, 8}) {
            const int iters = 400;
            int64_t bytes = 0;
            const double t0 = now_ms();
            for (int it = 0; it < iters; ++it) {
                const int il = it % n_layer;
                int ids[8];
                for (int j = 0; j < n; ++j) ids[j] = (it * 5 + j * 2 + 1) % test_experts;
                moe.compute_layer_sync(il, x.data(), ids, wts, n, out.data());
                bytes += descs[il].blob * n;
            }
            const double ms = now_ms() - t0;
            log("cpu moe  %d experts/layer, %d threads: %.1f us/layer, %.1f GB/s (hot-ish cache: %lld MB working set)",
                n, threads, ms * 1000 / iters, bytes / ms / 1e6, (long long)(test_experts * descs[0].blob * n_layer >> 20));
        }
    }
    moe.stop();
    }
    log(fails ? "FAILED (%d)" : "all passed", fails);
    return fails ? 1 : 0;
}
