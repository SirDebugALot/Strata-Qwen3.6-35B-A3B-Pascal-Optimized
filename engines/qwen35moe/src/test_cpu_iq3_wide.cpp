// Production wide+PF2048 versus unchanged nibble and native IQ3 dots. CPU only.
// Optional model argument also exercises real CpuIq3DownCache storage/strides for experts0/17/255.
#include "common.hpp"
#include "cpu_iq.hpp"
#include "cpu_iq3_cache.hpp"
#include "cpu_iq3_nibble.hpp"
#include "gguf.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
using Block = sq::native_iq::Iq3NibbleBlock;
constexpr float Sentinel = 123456.25f;
constexpr int Guard = 8;
uint64_t checks = 0, cases = 0;

void compare(const Block* weights, const uint8_t* native, int rows, int nb, const char* name) {
    std::array<std::vector<sq::Q8K>, 4> storage;
    std::mt19937 rng(61008 + rows + nb);
    const int8_t edges[] = {-128,127,-127,0,1,-1,64,-64};
    for (int t = 0; t < 4; ++t) {
        storage[t].resize(nb);
        for (int b = 0; b < nb; ++b) {
            storage[t][b].d = ((t + b) & 1 ? -1.f : 1.f) * (t + 1) / 256.f;
            for (int k = 0; k < 256; ++k)
                storage[t][b].nat[k] = k % 3 ? edges[(k + 3 * t + b) % 8] : (int8_t)rng();
        }
    }
    for (int nt = 1; nt <= 4; ++nt) for (int pattern = 0; pattern < 3; ++pattern) {
        // Full, gapped/repeated token mapping and empty-range cases retain untouched output guards.
        const int map[3][4] = {{0,1,2,3}, {3,1,3,0}, {2,2,0,1}};
        const int r0 = pattern ? 3 : 0;
        const int r1 = pattern == 2 ? r0 : pattern == 1 ? rows - 2 : rows;
        std::array<std::vector<float>, 4> old, wide;
        const sq::Q8K* acts[4]; float* old_out[4]; float* wide_out[4];
        for (int t = 0; t < 4; ++t) {
            old[t].assign(rows + 2 * Guard, Sentinel); wide[t] = old[t];
            acts[t] = storage[map[pattern][t]].data();
            old_out[t] = old[t].data() + Guard; wide_out[t] = wide[t].data() + Guard;
        }
        sq::native_iq::iq3_nibble_rows(weights, nb, acts, nt, old_out, r0, r1);
        sq::native_iq::iq3_nibble_rows_wide_prefetch(weights, nb, acts, nt, wide_out, r0, r1);
        for (int t = 0; t < 4; ++t) for (size_t i = 0; i < old[t].size(); ++i) {
            SQ_CHECK(std::isfinite(old[t][i]) && std::isfinite(wide[t][i]) &&
                         std::memcmp(&old[t][i], &wide[t][i], sizeof(float)) == 0,
                     "%s NT%d pattern%d token%d index%zu old/new differ", name, nt, pattern, t, i);
            const int row = (int)i - Guard;
            const bool active = t < nt && row >= r0 && row < r1;
            if (!active) SQ_CHECK(old[t][i] == Sentinel, "inactive output/guard changed");
            else if (native) {
                const float reference = sq::native_iq::dot_iq3s(native + (size_t)row * nb * 110, acts[t], nb);
                SQ_CHECK(std::isfinite(reference) && std::memcmp(&reference, &wide[t][i], sizeof(float)) == 0,
                         "%s NT%d pattern%d token%d row%d native oracle differs", name, nt, pattern, t, row);
                ++checks;
            }
            ++checks;
        }
        ++cases;
    }
}

void synthetic() {
    for (int nb : {1, 2, 8}) {
        constexpr int Rows = 512;
        std::vector<uint8_t> native((size_t)Rows * nb * 110);
        std::mt19937 rng(61124 + nb);
        for (auto& b : native) b = (uint8_t)rng();
        for (int row = 0; row < Rows; ++row) for (int i = 0; i < nb; ++i) {
            uint8_t* block = native.data() + ((size_t)row * nb + i) * 110;
            const uint16_t scale = sq::f32_to_fp16((row & 1 ? -1.f : 1.f) / 256.f);
            std::memcpy(block, &scale, 2);
            for (int h = 0; h < 8; ++h) {
                block[66 + h] = 0;
                for (int g = 0; g < 8; ++g) {
                    const int grid = (row + i + 17 * h + 31 * g) & 511;
                    block[2 + 8 * h + g] = (uint8_t)grid;
                    block[66 + h] |= (uint8_t)((grid >> 8) << g);
                }
                for (int b = 0; b < 4; ++b) block[74 + 4 * h + b] = (uint8_t)(row + 23 * h + 47 * b);
            }
            for (int b = 0; b < 4; ++b) block[106 + b] = (uint8_t)(row + 63 * b);
        }
        std::vector<Block> decoded((size_t)Rows * nb);
        sq::native_iq::decode_iq3_nibbles(native.data(), decoded.data(), decoded.size());
        compare(decoded.data(), native.data(), Rows, nb, "all512-grids/signs/scales");
        // Every possible packed byte at every coefficient position across the512 rows.
        for (int row = 0; row < Rows; ++row) for (int b = 0; b < nb; ++b)
            for (int c = 0; c < 128; ++c)
                decoded[(size_t)row * nb + b].coefficients[c] = (uint8_t)(row + 17 * c + 31 * b);
        compare(decoded.data(), nullptr, Rows, nb, "all256-packed-byte-pairs");
    }
}

void real_cache(const char* path) {
    sq::GgufFile model; std::string error;
    SQ_CHECK(model.open(path, error), "%s", error.c_str());
    for (int layer = 0; layer < 40; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto& down = model.need(prefix + "ffn_down_exps.weight");
        if (down.type != sq::T_IQ3_S) continue;
        const auto& gate = model.need(prefix + "ffn_gate_exps.weight");
        const auto& up = model.need(prefix + "ffn_up_exps.weight");
        SQ_CHECK(gate.type == up.type && gate.ne == up.ne && gate.ne == std::vector<int64_t>({2048,512,256}) &&
                     down.ne == std::vector<int64_t>({512,2048,256}), "unexpected GGUF expert shapes");
        const size_t gu = (size_t)sq::row_bytes(gate.type, 2048) * 512;
        const size_t down_bytes = (size_t)2048 * 2 * 110, blob = 2 * gu + down_bytes;
        std::vector<uint8_t> originals(3 * blob);
        const int ids[] = {0,17,255};
        for (int i = 0; i < 3; ++i) {
            uint8_t* target = originals.data() + i * blob;
            std::memcpy(target, gate.data + ids[i] * gu, gu);
            std::memcpy(target + gu, up.data + ids[i] * gu, gu);
            std::memcpy(target + 2 * gu, down.data + ids[i] * down_bytes, down_bytes);
        }
        sq::ExpertLayerDesc descriptor{originals.data(), (int64_t)blob, (int64_t)gu, gate.type, down.type, 3};
        sq::CpuIq3DownCache cache;
        SQ_CHECK(cache.build({descriptor}, 3, error), "%s", error.c_str());
        SQ_CHECK(cache.bytes() == 3 * sq::CpuIq3DownCache::BlocksPerExpert * sizeof(Block), "unexpected cache allocation");
        for (int i = 0; i < 3; ++i)
            compare(cache.layer(0) + (size_t)i * sq::CpuIq3DownCache::BlocksPerExpert,
                    originals.data() + i * blob + 2 * gu, 2048, 2, "real-cache-expert");
        return;
    }
    sq::die("model has no IQ3_S down layer");
}
}

int main(int argc, char** argv) {
    SQ_CHECK(argc <= 2, "usage: test_cpu_iq3_wide [model.gguf]");
    SQ_CHECK(sq::cpu_force_isa("avx2"), "AVX2 required");
    const bool setting = sq::native_iq::iq3_nibble_wide_enabled(); // validates the actual cached production parser
    synthetic();
    if (argc == 2) real_cache(argv[1]);
    std::printf("CPU IQ3 WIDE EXACT PASS cases=%llu float/guard/native checks=%llu configured=%d NT1..4 all512-grids/all256-packed/signed128/gapped/repeated\n",
                (unsigned long long)cases, (unsigned long long)checks, setting ? 1 : 0);
    return 0;
}
