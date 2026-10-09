#include "cpu_iq3_cache.hpp"
#include "cpu_moe.hpp"
#include "common.hpp"
#include <algorithm>
#include <atomic>
#include <limits>
#include <new>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace sq {
namespace {
constexpr size_t NativeExpertBytes = CpuIq3DownCache::BlocksPerExpert * 110;
constexpr uint64_t Headroom = 4ull * 1024 * 1024 * 1024;
struct Memory { uint64_t physical = 0, commit = 0; bool commit_known = false; };
bool available_memory(Memory& m) {
#ifdef _WIN32
    MEMORYSTATUSEX status{}; status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    m.physical = status.ullAvailPhys;
    m.commit = status.ullAvailPageFile;
    m.commit_known = true;
#else
    const long pages = sysconf(_SC_AVPHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) return false;
    m.physical = (uint64_t)pages * (uint64_t)page_size;
#endif
    return true;
}
}

bool CpuIq3DownCache::estimate_bytes(const std::vector<ExpertLayerDesc>& layers, size_t& bytes, std::string& error) {
    bytes = 0;
    error.clear();
    constexpr size_t ExpandedExpertBytes = BlocksPerExpert * sizeof(native_iq::Iq3NibbleBlock);
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = layers[i];
        if (layer.t_down != T_IQ3_S) continue;
        if (!layer.base || layer.n_expert <= 0 || layer.gu_bytes <= 0 ||
            layer.gu_bytes > (std::numeric_limits<int64_t>::max() - (int64_t)NativeExpertBytes) / 2 ||
            layer.blob < 2 * layer.gu_bytes + (int64_t)NativeExpertBytes ||
            (size_t)layer.n_expert > std::numeric_limits<size_t>::max() / (size_t)layer.blob) {
            error = "invalid IQ3_S layer descriptor at layer " + std::to_string(i);
            return false;
        }
        if ((size_t)layer.n_expert > (std::numeric_limits<size_t>::max() - bytes) / ExpandedExpertBytes) {
            error = "IQ3_S allocation size overflow";
            return false;
        }
        bytes += (size_t)layer.n_expert * ExpandedExpertBytes;
    }
    return true;
}

bool CpuIq3DownCache::build(const std::vector<ExpertLayerDesc>& source, int workers, std::string& error) {
    size_t required = 0;
    if (!estimate_bytes(source, required, error)) return false;
    if (required && !cpu_has_avx2()) { error = "AVX2/F16C/FMA are required"; return false; }
    Memory memory;
    if (required && (!available_memory(memory) || required > std::numeric_limits<uint64_t>::max() - Headroom ||
                     memory.physical < (uint64_t)required + Headroom ||
                     (memory.commit_known && memory.commit < (uint64_t)required + Headroom))) {
        error = "need " + std::to_string(required) + " bytes plus 4 GiB physical/commit headroom; available physical=" +
                std::to_string(memory.physical) + ", commit=" + (memory.commit_known ? std::to_string(memory.commit) : "unknown");
        return false;
    }
    // Candidate arrays are not published until every conversion worker has joined. Original pinned/native
    // blobs remain untouched and remain the sole source for GPU uploads; these new arrays are ordinary RAM.
    struct Job { const uint8_t* source; native_iq::Iq3NibbleBlock* destination; };
    std::vector<Layer> candidate;
    std::vector<Job> jobs;
    const double start = now_ms();
    try {
        candidate.resize(source.size());
        for (size_t l = 0; l < source.size(); ++l) {
            const auto& layer = source[l];
            if (layer.t_down != T_IQ3_S) continue;
            const size_t blocks = (size_t)layer.n_expert * BlocksPerExpert;
            // Deliberately no value initialization: conversion writes every byte before publication.
            candidate[l].data.reset(new (std::nothrow) native_iq::Iq3NibbleBlock[blocks]);
            if (!candidate[l].data) { error = "pageable IQ3_S allocation failed at layer " + std::to_string(l); return false; }
            for (int e = 0; e < layer.n_expert; ++e)
                jobs.push_back({layer.base + (size_t)e * (size_t)layer.blob + 2 * (size_t)layer.gu_bytes,
                                candidate[l].data.get() + (size_t)e * BlocksPerExpert});
        }
    } catch (const std::exception& exception) { error = exception.what(); return false; }
    std::atomic<size_t> next{0};
    const int count = jobs.empty() ? 0 : std::min({std::max(workers, 1), 6, (int)jobs.size()});
    const auto convert = [&] {
        for (;;) {
            const size_t index = next.fetch_add(1, std::memory_order_relaxed);
            if (index >= jobs.size()) break;
            native_iq::decode_iq3_nibbles(jobs[index].source, jobs[index].destination, BlocksPerExpert);
        }
    };
    std::vector<std::thread> builders;
    try {
        builders.reserve(count);
        for (int t = 0; t < count; ++t) builders.emplace_back(convert);
    } catch (const std::exception& exception) {
        // A constructor may fail after earlier threads started. They finish their independent jobs before
        // candidate storage is destroyed; none can outlive this function, even on failure.
        for (auto& thread : builders) thread.join();
        error = std::string("IQ3_S conversion worker startup failed: ") + exception.what();
        return false;
    }
    for (auto& thread : builders) thread.join();
    layers_ = std::move(candidate);
    bytes_ = required;
    size_t eligible = 0;
    for (const auto& layer : layers_) if (layer.data) ++eligible;
    log("CPU IQ3 nibble down cache: %zu layers, %zu experts, %zu pageable bytes (%.2f MiB), %d build workers, %.1f ms; original GPU blobs unchanged",
        eligible, jobs.size(), bytes_, bytes_ / 1048576.0, count, now_ms() - start);
    return true;
}
}
