// cpu_moe.hpp - the CPU half of the MoE: routed experts that are not resident in VRAM are computed here, in
// place in pinned RAM, while the GPU computes the resident ones.
//
// Hand-off with the GPU is through host-mapped memory, one mailbox per layer:
//   GPU router kernel  -> Mailbox{x, miss ids, weights}, then ready_seq = seq     (after __threadfence_system)
//   CPU workers        -> Result{out}, then done_seq = seq                        (after a release fence)
//   GPU combine kernel spins on done_seq when the layer had misses.
// The opt-in event path instead copies a device mailbox to pinned RAM, waits for its CUDA event on the host,
// and uses begin_layer/end_layer.  The engine copies the completed result back before launching the combine.
#pragma once

#include "quant.hpp"

#include <atomic>
#include <memory>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace sq {

// ------------------------------------------------------------------ activation format for the CPU dot products
/// 256 activations quantized to int8 with one float scale.  `nat` is natural order; `pair` is the order the
/// Q4_K/Q5_K nibble layout wants: [0..31, 64..95 | 32..63, 96..127 | 128..159, 192..223 | 160..191, 224..255].
struct alignas(64) Q8K {
    int8_t nat[256];
    int8_t pair[256];
    int16_t bsums[16];   // sums of each 16 consecutive (natural order) quants
    float d;
    int32_t _pad[7];
};

void quantize_q8k(const float* x, Q8K* y, int n);
/// Dot product of `nb` super-blocks of one weight row against `a` (nb entries).
float dot_row(uint32_t type, const void* w, const Q8K* a, int nb);
/// The row dot product of `type` (Q4_K, Q5_K, Q6_K; else the scalar reference), with the kernels in use.
using DotFn = float (*)(const void* w, const Q8K* a, int nb);
DotFn dot_fn(uint32_t type);
/// The instruction set the CPU kernels use: "AVX-512 VNNI" where the CPU and OS have it, else "AVX2".  The first
/// call decides; STRATA_CPU_ISA=avx2 (or cpu_force_isa) picks the AVX2 kernels on any CPU.
const char* cpu_isa();
/// "avx512" or "avx2"; false if this CPU cannot run it.  Before the kernels are used.
bool cpu_force_isa(const char* name);
/// True when the CPU runs the AVX2 kernels (AVX2, FMA, F16C and the OS saving the YMM state).
bool cpu_has_avx2();
/// Scalar reference: dequantize the row and dot with the dequantized activation.
float dot_row_ref(uint32_t type, const void* w, const Q8K* a, int nb);

// ------------------------------------------------------------------ mailboxes (host-mapped)
constexpr int kMaxK = 8;             // experts per token
constexpr int kMaxT = 4;             // tokens per step (1 = plain decode, more = speculative verification / MTP)
constexpr int kMaxU = kMaxK * kMaxT; // distinct experts per layer and step

/// One layer's missing experts for a step of n_tok tokens: the distinct experts, and for each of them the routing
/// weight of every token (0 = that token did not pick it).  x holds the tokens' normalized hidden states.
struct alignas(64) Mailbox {
    volatile uint32_t ready_seq;
    uint32_t _p0[15];
    int32_t n_miss;
    int32_t n_tok;
    int32_t _p1[14];
    int32_t ids[kMaxU];
    float w[kMaxU][kMaxT];
    float x[kMaxT][2048];
};
static_assert(sizeof(Mailbox) % 64 == 0);

struct alignas(64) Result {
    volatile uint32_t done_seq;
    uint32_t _p0[15];
    float out[kMaxT][2048];
};

struct ExpertLayerDesc {
    const uint8_t* base = nullptr;   // pinned blobs
    int64_t blob = 0, gu_bytes = 0;
    uint32_t t_gu = 0, t_down = 0;
    int n_expert = 0; // explicit source extent for optional CPU-only derived formats
};

// Diagnostic summed worker elapsed durations (including waiting/preemption), not scheduled CPU time.
// Only the host merges these after end_layer's
// acquire. The fused legacy down path is reported separately rather than timing every individual dot.
enum CpuMoePhase { CpuPackGU, CpuGU, CpuGUBarrier, CpuPackDown, CpuDown, CpuReduce, CpuDownFused,
                   CpuWorkerTotal, CpuPhaseCount };
struct CpuMoePhaseCounts {
    uint64_t ns[CpuPhaseCount] = {};
    uint64_t gu_quant_tokens = 0, down_quant_pairs = 0;
    uint64_t gu_row_pairs = 0, down_row_pairs = 0;
};
struct CpuMoePhaseGroup {
    int n_tok = 0;
    uint32_t t_gu = 0, t_down = 0;
    uint64_t jobs = 0, missing_experts = 0, routed_pairs = 0;
    uint64_t expert_nt[kMaxT + 1] = {}; // includes NT0 (zero routing weight), distinct missing experts only
    uint64_t dispatch_to_join_ns = 0, max_worker_ns = 0;
    CpuMoePhaseCounts workers;
};
struct CpuMoePhaseProfile {
    uint64_t seen_jobs = 0, sampled_jobs = 0;
    bool complete = false;
    std::vector<CpuMoePhaseGroup> groups;
};

class CpuIq3DownCache;
class CpuMoe {
public:
    CpuMoe();
    ~CpuMoe();
    /// `mb`/`res` are host pointers to n_layer mailboxes/results (mapped memory).
    void start(int n_threads, int n_layer, const std::vector<ExpertLayerDesc>& layers, Mailbox* mb, Result* res,
               bool pin_threads);
    void stop();
    int threads() const { return n_threads_; }
    size_t iq3_down_cache_bytes() const;
    /// Read only after end_step/end_layer: each worker owns its counter while a job runs.
    uint64_t iq3_down_cache_rows() const;
    /// Host-only diagnostic snapshot; read between drained jobs. Only begin_layer/end_layer jobs are sampled.
    const CpuMoePhaseProfile& phase_profile() const { return phase_profile_; }

    /// Run one callback per persistent worker and wait for completion. Called by the sole host scheduler,
    /// between MoE jobs; callbacks must not dispatch nested work or throw. Worker ids are 0..threads()-1.
    void parallel_for_workers(std::function<void(int)> job);

    /// Arms the workers for one step: they walk layers l0..l1-1 waiting for the GPU's mailboxes.
    void begin_step(uint32_t seq, int l0, int l1);
    /// Blocks until the workers finished the step.  Returns false if they timed out waiting for the GPU.
    bool end_step();

    /// Host/event handoff: the mailbox must already be complete, and remain unchanged until end_layer().
    /// Dispatches one layer without waiting, so the GPU can compute its resident experts concurrently.
    void begin_layer(int layer);
    /// Waits for every worker; result rows are then safe to copy to the GPU. Supports 1..kMaxT tokens.
    /// prompt selects the immutable startup prompt wait budget; omitted retains the ordinary budget.
    void end_layer(bool prompt = false);

    /// Synchronous single-layer compute (tests, fallbacks).  out = sum_j w_j * expert_j(x).
    void compute_layer_sync(int layer, const float* x, const int* ids, const float* w, int n, float* out);

    uint64_t experts_computed = 0;

private:
    void dispatch(std::function<void(int)> job);
    void wait_job(bool prompt = false);
    void worker(int tid, uint32_t gen0);
    /// Returns true on the thread that finished the layer's last output rows.  x / out: n_tok rows of 2048.
    bool run_layer(int tid, int layer, const float (*x)[2048], const int* ids, const float (*w)[kMaxT], int n, int n_tok,
                   float (*out)[2048]);
    template<bool Profile> bool run_layer_impl(int tid, int layer, const float (*x)[2048], const int* ids,
                                              const float (*w)[kMaxT], int n, int n_tok, float (*out)[2048]);
    void reset_sync();
    void phase_begin_layer(int layer);
    void phase_end_layer();
    void phase_log(bool partial);

    int n_threads_ = 0, n_layer_ = 0;
    int worker_spins_ = 200000; // STRATA_CPU_WORKER_SPINS, read once before workers start; 0 sleeps immediately
    int completion_spins_ = -1; // STRATA_CPU_COMPLETION_SPINS: negative=legacy; otherwise pause then atomic-wait
    int prompt_completion_spins_ = -1; // STRATA_PROMPT_COMPLETION_SPINS: missing inherits ordinary budget
    bool notify_completion_ = false; // immutable while workers run: either possible waiter can sleep
    bool phase_enabled_ = false, phase_active_ = false, phase_logged_ = false;
    int phase_layer_ = -1;
    uint64_t phase_skip_ = 40, phase_limit_ = 400, phase_start_ns_ = 0;
    struct alignas(64) PhaseWorker { CpuMoePhaseCounts counts; };
    std::unique_ptr<PhaseWorker[]> phase_workers_; // allocated only for opt-in diagnostics, worker-private
    CpuMoePhaseProfile phase_profile_; // host-only, never read or written by workers
    std::vector<ExpertLayerDesc> layers_;
    std::unique_ptr<CpuIq3DownCache> iq3_down_cache_; // immutable after start; freed after worker join
    Mailbox* mb_ = nullptr;
    Result* res_ = nullptr;
    std::vector<std::thread> threads_;

    std::function<void(int)> job_;
    std::atomic<uint32_t> job_gen_{0};
    std::atomic<int> job_left_{0};
    std::atomic<bool> quit_{false};
    std::atomic<bool> timed_out_{false};
    // Work is handed out in row chunks through per-layer counters, not split statically: a worker the OS preempts
    // only delays the chunk it holds, and the others finish the layer without it.
    struct alignas(64) Counter { std::atomic<int> v{0}; };
    struct LayerSync { Counter take1, done1, take2, done2; };
    std::unique_ptr<LayerSync[]> sync_;         // per layer, reset before each step

    static constexpr int kFF = 512, kE = 2048;
    static constexpr int kChunkGU = 32, kChunkD = 32;   // rows per work item (gate+up / down)
    struct alignas(64) Scratch {
        Q8K xq[kMaxT][kE / 256];
        Q8K hq[kMaxU][kFF / 256];
        uint64_t nibble_rows = 0;
    };
    std::vector<Scratch> scratch_;               // per thread
    alignas(64) float h_[kMaxU][kFF];            // shared: gate/up output per (expert, token) pair
};

}  // namespace sq
