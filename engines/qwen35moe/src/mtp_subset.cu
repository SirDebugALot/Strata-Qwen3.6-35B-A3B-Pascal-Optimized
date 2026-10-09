// Dynamic MTP draft head. The dense row arithmetic below deliberately matches kernels.cu gemv_q8_kernel<1>.
#include "mtp_subset.cuh"
#include "actq_readonly.cuh"
#include "common.hpp"
#include "dense_q6.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>

namespace sq {
namespace {

// Stratified top-K, not a global top-K: retaining candidates from every vocabulary region also covers scripts
// far from the English token range. Recent tokens are appended separately. Padded IDs never reach the head.
__global__ void select_partition(const float* logits, int vocab, int keep, int32_t* ids) {
    using Sort = cub::BlockRadixSort<float, 256, 8, int32_t>;
    __shared__ typename Sort::TempStorage scratch;
    const int span = (vocab + MtpDraftSubset::partitions - 1) / MtpDraftSubset::partitions;
    const int begin = blockIdx.x * span;
    float value[8];
    int32_t id[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int offset = threadIdx.x * 8 + i;
        const int token = begin + offset;
        const bool valid = offset < span && token < vocab;
        value[i] = valid ? logits[token] : -CUDART_INF_F;
        id[i] = valid ? token : -1;
    }
    Sort(scratch).SortDescendingBlockedToStriped(value, id);
    if (threadIdx.x < keep) ids[blockIdx.x * keep + threadIdx.x] = id[0];
}

__global__ void first_occurrence(const int32_t* ids, int count, int vocab, int32_t* first) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < count) {
        const int token = ids[slot];
        if (token >= 0 && token < vocab) atomicMin(first + token, slot);
    }
}

constexpr int kWarps = 8;

template<bool ReadOnly = false>
__global__ void __launch_bounds__(kWarps * 32)
gather_head(const int8_t* __restrict__ wq, const __half* __restrict__ wd, ActQ a,
            const int32_t* ids, const int32_t* first, float* __restrict__ y,
            int slots, int vocab, int cols) {
    extern __shared__ int4 smem[];
    const int nvec = cols >> 4, nsc = cols >> 5;
    int4* sa = smem;
    float* sd = (float*)(smem + nvec);
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) sa[i] = actq_load<ReadOnly>((const int4*)a.q + i);
    for (int i = threadIdx.x; i < nsc; i += blockDim.x) sd[i] = actq_load<ReadOnly>(a.d + i);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int slot = blockIdx.x * kWarps + warp;
    if (slot >= slots) return;
    const int row = ids[slot];
    if (row < 0 || row >= vocab || first[row] != slot) {
        if (lane == 0) y[slot] = -CUDART_INF_F;
        return;
    }
    const int4* wr = (const int4*)(wq + (size_t)row * cols);
    const __half* wdr = wd + (size_t)row * nsc;
    float acc = 0.0f;
#pragma unroll 4
    for (int i = lane; i < nvec; i += 32) {
        const int4 w4 = __ldg(wr + i);
        const float ws = __half2float(wdr[i >> 1]);
        const int4 a4 = sa[i];
        int s = __dp4a(w4.x, a4.x, 0);
        s = __dp4a(w4.y, a4.y, s);
        s = __dp4a(w4.z, a4.z, s);
        s = __dp4a(w4.w, a4.w, s);
        acc += (float)s * ws * sd[i >> 1];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    if (lane == 0) y[slot] = acc;
}

__global__ void map_token(const int32_t* ids, int32_t* token) { *token = ids[*token]; }

} // namespace

MtpDraftSubset::MtpDraftSubset(int vocab, int per_partition) : vocab_(vocab), per_partition_(per_partition) {
    SQ_CHECK(vocab > 0 && vocab <= partitions * 2048, "MTP subset supports 1..%d vocabulary entries", partitions * 2048);
    SQ_CHECK(per_partition == 32 || per_partition == 64 || per_partition == 128,
             "MTP subset per-partition size must be 32, 64, or 128");
    CUDA_CHECK(cudaMalloc(&ids_, (size_t)slots() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&first_slot_, (size_t)vocab * sizeof(int32_t)));
    CUDA_CHECK(cudaHostAlloc(&recent_host_, recent_slots * sizeof(int32_t), cudaHostAllocDefault));
}

MtpDraftSubset::~MtpDraftSubset() {
    // The owner drains its stream before destruction; do not release the pinned source while its copy is pending.
    if (ids_) cudaFree(ids_);
    if (first_slot_) cudaFree(first_slot_);
    if (recent_host_) cudaFreeHost(recent_host_);
}

void MtpDraftSubset::prepare(const float* logits, const std::vector<int>& history, int next, cudaStream_t stream) {
    std::fill_n(recent_host_, recent_slots, -1);
    const int n = (int)std::min(history.size(), (size_t)recent_slots - 1);
    for (int i = 0; i < n; ++i) recent_host_[i] = history[history.size() - n + i];
    recent_host_[n] = next;
    select_partition<<<partitions, 256, 0, stream>>>(logits, vocab_, per_partition_, ids_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(ids_ + partitions * per_partition_, recent_host_, recent_slots * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(first_slot_, 0x7f, (size_t)vocab_ * sizeof(int32_t), stream));
    first_occurrence<<<(slots() + 255) / 256, 256, 0, stream>>>(ids_, slots(), vocab_, first_slot_);
    CUDA_CHECK(cudaGetLastError());
}

void MtpDraftSubset::project(const DQ8& head, ActQ activation, float* logits, int32_t* token, float* prob,
                             cudaStream_t stream, int readonly_override) const {
    SQ_CHECK(readonly_override >= -1 && readonly_override <= 1, "invalid ActQ read-only test override");
    const bool readonly = readonly_override < 0 ? actq_readonly_enabled() : readonly_override != 0;
    SQ_CHECK(head.rows == vocab_ && head.cols % 32 == 0, "MTP subset head shape mismatch");
    if (head.q6) {
        k_gather_q6(head, activation, ids_, first_slot_, slots(), logits, stream);
    } else {
        const size_t shared = (size_t)head.cols + (size_t)(head.cols / 32) * sizeof(float);
        if (readonly) {
            gather_head<true><<<(slots() + kWarps - 1) / kWarps, kWarps * 32, shared, stream>>>(
                head.qs, (const __half*)head.d, activation, ids_, first_slot_, logits, slots(), vocab_, head.cols);
        } else {
            gather_head<false><<<(slots() + kWarps - 1) / kWarps, kWarps * 32, shared, stream>>>(
                head.qs, (const __half*)head.d, activation, ids_, first_slot_, logits, slots(), vocab_, head.cols);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    k_argmax(logits, slots(), token, 1, stream, prob);
    map_token<<<1, 1, 0, stream>>>(ids_, token);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sq
