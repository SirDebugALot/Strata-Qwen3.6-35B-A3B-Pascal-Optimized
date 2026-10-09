// Optional dynamic draft vocabulary. Target verification always keeps the full vocabulary.
#pragma once

#include "kernels.cuh"

#include <vector>

namespace sq {

class MtpDraftSubset {
public:
    static constexpr int partitions = 128;
    static constexpr int recent_slots = 256;
    MtpDraftSubset(int vocab, int per_partition);
    ~MtpDraftSubset();
    MtpDraftSubset(const MtpDraftSubset&) = delete;
    MtpDraftSubset& operator=(const MtpDraftSubset&) = delete;

    // Once per speculative round, before the target logits can be overwritten. The stream must finish using
    // the previous preparation before this method is called again (Engine::run_step already waits for it).
    void prepare(const float* target_logits, const std::vector<int>& history, int next, cudaStream_t stream);
    // Gather rows directly from the original head: no copied head weights or changes to their arithmetic.
    // prob is normalized over this subset and is only a draft-depth heuristic, never a proposal probability.
    void project(const DQ8& head, ActQ activation, float* logits, int32_t* token, float* prob,
                 cudaStream_t stream, int readonly_override = -1) const;
    // readonly_override=-1 uses the process flag; 0/1 are explicit test variants.

    int slots() const { return partitions * per_partition_ + recent_slots; }
    int per_partition() const { return per_partition_; }
    const int32_t* ids_device() const { return ids_; }
    bool active = false; // scoped to Engine::spec_step; ordinary prompt catch-up retains its original path

private:
    int vocab_, per_partition_;
    int32_t* ids_ = nullptr;
    int32_t* first_slot_ = nullptr;
    int32_t* recent_host_ = nullptr;
};

} // namespace sq
