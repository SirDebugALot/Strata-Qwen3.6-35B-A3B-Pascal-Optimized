// engine.hpp - the inference engine: state (KV cache, delta-net states), the VRAM expert cache, the decode step
// (one CUDA graph per step, CPU experts in parallel), prefill, prefix reuse, and speculative decoding with the
// model's MTP (multi-token prediction) block.
//
// MTP: the MTP block predicts token i+2 from the trunk's normed hidden state h_i and the embedding of token i+1.
// A speculative step drafts D tokens with it (the first from the pending (h, token) pairs, the rest recursively
// from its own hidden state), verifies [token, drafts] in one decode step of D+1 tokens, keeps the longest accepted
// prefix plus one token sampled by the trunk, and rolls the delta-net state back to the last accepted token (its
// state after every token of the step is snapshotted).  The MTP layer has its own KV cache, filled for every
// position from (h_i, token i+1): during prefill for the prompt, and at the start of each step for the tokens the
// previous step accepted ("pending pairs": their hidden states wait in mtp_hin_ until the next token is known).
#pragma once

#include "common.hpp"
#include "cpu_moe.hpp"
#include "kernels.cuh"
#include "model.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace sq {

struct EngineOptions {
    std::string model_path;
    int ctx = 32768;
    int cpu_threads = 8;
    int64_t vram_reserve_mb = 1200;  // left free on the card once the engine is ready (other programs, driver)
    int64_t cache_mb = -1;           // expert cache size; -1 = everything that is free
    std::string profile_path;        // expert routing profile used to pick the resident experts
    bool use_graph = true;
    bool pin_threads = true;
    int prefill_chunk = 4096;        // 0 = decode-only prompts, no batched-prefill workspaces
    int prefill_min = 8;             // shorter inputs go through the decode path
    int adapt_every = 32;            // decode steps between cache adaptation rounds (0 = off)
    int adapt_swaps = 24;            // max experts swapped in per round
    double adapt_decay = 0.97;       // retain this fraction of routing frequency per adaptation round
    double profile_weight = 1.0;    // loaded profile vs live routing counts (see setup_cache)
    int ckpt_slots = 3;              // prefix-reuse checkpoints (~66 MiB each); 0 = no checkpoints
    int mtp_draft = 3;               // speculative tokens drafted per step by the MTP block (0 = off; max kMaxT - 1)
    float draft_p = 0.8f;            // keep drafting while the MTP's probability of all drafts so far is >= this
                                     // (half of it for the first draft; below that the step is a plain one)
    bool verbose = false;
};

struct SamplingParams {
    float temperature = 0.0f;   // 0 = greedy
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float repetition_penalty = 1.0f;
    int penalty_last_n = 64;    // the penalties count the last N tokens (prompt and output), as llama.cpp does
    uint64_t seed = 0;
    bool greedy() const {
        return temperature <= 0.0f && presence_penalty == 0.0f && frequency_penalty == 0.0f && repetition_penalty == 1.0f;
    }
    bool penalized() const { return presence_penalty != 0.0f || frequency_penalty != 0.0f || repetition_penalty != 1.0f; }
};

struct EngineStats {
    uint64_t decode_steps = 0;               // tokens fed through the decode path (a speculative step feeds several)
    uint64_t graph_steps = 0;                // decode-step graph launches (trunk)
    double decode_ms = 0, prefill_ms = 0;
    double adapt_ms = 0;                     // expert cache re-ranking + swaps after decode steps
    uint64_t prefill_tokens = 0;
    uint64_t swaps = 0;
    uint64_t streamed_experts = 0, streamed_pairs = 0, streamed_bytes = 0; // transient misses, never cache hits
    uint64_t stream_max_unique = 0, stream_max_pairs = 0, stream_zero_cpu_layers = 0;
    uint64_t cache_rebalances = 0, rebalance_slots = 0, rebalance_bytes = 0;
    double rebalance_ms = 0;                  // also included in adapt_ms, not additional decode time
    uint64_t spec_steps = 0, drafted = 0, accepted = 0;   // MTP speculative steps, drafts, accepted drafts
    uint64_t spec_skipped = 0;               // steps where the first draft was too unlikely to verify
    double mtp_ms = 0;                       // time in the MTP block (drafting + pending pairs)
    double verify_ms = 0;                    // time in verification steps (trunk over token + drafts)
    std::vector<float> step_ms;              // recent per-token decode times (ring), for a noise-robust median
};

class Prefill;
class MtpDraftSubset;

class Engine {
public:
    Engine();
    ~Engine();
    bool init(const EngineOptions& opt, std::string& err);

    const Config& cfg() const { return model_.cfg; }
    int ctx() const { return opt_.ctx; }
    int n_past() const { return n_past_; }
    const std::vector<int>& tokens() const { return history_; }

    /// Forget everything (positions restart at 0).
    void reset();
    /// Feed tokens (appending to the current sequence).  Leaves the logits of the LAST token ready.
    void feed(const int* tokens, int n);
    /// Sample the next token from the current logits.
    int sample(const SamplingParams& sp, std::mt19937_64& rng);
    /// Start of a request (after the prompt is fed): the penalty window starts with the prompt's last tokens.
    void begin_request(const SamplingParams& sp);

    /// True when the MTP block is loaded and drafting is on.
    bool mtp() const { return mtp_on_; }
    int mtp_draft() const { return mtp_on_ ? draft_ : 0; }
    /// Drafts per step from now on (0 = plain decoding; at most the --mtp value the engine started with).
    void set_mtp_draft(int d) { draft_ = std::clamp(d, 0, opt_.mtp_draft); }
    /// One speculative step.  `tok` is the token just sampled (not fed yet).  Feeds it plus the accepted drafts and
    /// returns in `out` the tokens that follow `tok`: the accepted drafts (fed) and last, one token sampled from the
    /// trunk (not fed, like the result of sample()).  Without MTP: feeds `tok` and samples one token.
    /// A stop token or the last of `max_output` outputs is returned but never fed, so history remains the prefix
    /// actually emitted by the caller.  max_output must be positive; omitted limits preserve unrestricted steps.
    void spec_step(int tok, const SamplingParams& sp, std::mt19937_64& rng, std::vector<int>& out,
                   const std::vector<int>* stop_tokens = nullptr, int max_output = INT_MAX);

    /// Prefix reuse: how many leading tokens of `prompt` can be kept (the rest must be fed).  Picks the longest of
    /// the live state and the checkpoints and restores it when it returns > 0.
    int reuse_prefix(const std::vector<int>& prompt);
    /// Remember the current state in a checkpoint slot (least recently used one).  The delta-net state cannot be
    /// rolled back, so the server checkpoints the positions a follow-up request is likely to resume from.
    /// `base`: the end of the system message, which new conversations share - it gets a slot of its own (with 2+
    /// slots) instead of being pushed out by the turns of one long conversation.
    void save_checkpoint(bool base = false);

    /// Expert profile (routing frequencies seen so far, including the loaded profile) -> file.
    bool save_profile(const std::string& path, std::string& err);

    EngineStats stats;
    const Model& model() const { return model_; }
    int64_t cache_slots() const { return cache_slots_total_; }
    int ckpt_slots() const { return (int)ckpts_.size(); }
    double cache_gib() const { return cache_bytes_ / 1073741824.0; }
    std::string status_line() const;
    /// Per-token expert routes found resident / missing from cache, since the start (T1..4 comparable).
    void expert_counts(uint64_t& resident, uint64_t& missing) const;
    int64_t expert_arena_bytes() const { return model_.expert_bytes; }
    const char* kv_type() const { return kv_int8_ ? "int8" : "fp16"; }
    int kv_capacity() const { return kv_capacity_; }
    int64_t kv_bytes() const {
        const auto& c = model_.cfg;
        return (int64_t)(c.n_attn_layers() + c.n_mtp) * 2 * c.n_head_kv * kv_capacity_ *
               (kv_int8_ ? c.head_dim + 4 : c.head_dim * 2);
    }
    int cpu_threads() const { return cpu_ ? cpu_->threads() : 0; }
    uint64_t event_graph_steps() const { return event_graph_steps_; }

    // --- debugging
    void debug_set_miss_stream_eligible(const std::vector<std::vector<int>>& ids); // drained fixed-placement override
    void debug_set_miss_stream_serialized(bool enabled); // identical placement, serialize transient work against CPU
    std::vector<int> debug_cpu_miss_ids(int layer); // last completed mailbox after optional stream partition
    bool debug_miss_stream_payloads_match(); // last occupied staging bank versus immutable original blobs
    const std::vector<std::vector<int>>& debug_miss_stream_eligible() const { return miss_stream_eligible_; }
    uint64_t debug_event_graph_captures() const { return event_graph_captures_; }
    uint64_t debug_direct_pinned_uploads() const { return direct_pinned_uploads_; }
    bool debug_cache_payloads_match(); // drain and compare every resident blob with its immutable model source
    void debug_set_mtp_last_row(bool enabled); // same-engine parity; init with STRATA_MTP_LAST_ROW=1 first
    uint64_t debug_mtp_last_row_steps() const { return mtp_last_row_steps_; }
    uint64_t debug_mtp_kv_only_steps() const { return mtp_kv_only_steps_; }
    uint64_t debug_prompt_heads_skipped() const { return prompt_heads_skipped_; }
    uint64_t debug_prompt_snapshot_bytes_skipped() const { return prompt_snapshot_bytes_skipped_; }
    uint64_t debug_prompt_mtp_submitted() const { return prompt_mtp_submitted_; }
    uint64_t debug_prompt_route_captures() const { return prompt_route_captures_; }
    uint64_t debug_prompt_route_launches() const { return prompt_route_launches_; }
    uint64_t debug_prompt_full_graph_captures() const { return prompt_full_captures_; }
    uint64_t debug_prompt_full_graph_launches() const { return prompt_full_launches_[0] + prompt_full_launches_[1]; }
    uint64_t debug_prompt_full_graph_variant_launches(bool skip_head) const { return prompt_full_launches_[skip_head ? 1 : 0]; }
    std::vector<float> debug_mtp_catch_up(int next, bool need_draft); // full MTP head, or KV-only state comparison
    void debug_freeze_cache_adaptation() { opt_.adapt_every = 0; } // retain current placement, including queued uploads
    int debug_prefill_chunk() const { return opt_.prefill_chunk; }
    int debug_set_prefill_chunk(int chunk) {
        SQ_CHECK(chunk > 0 && prefill_phase_switch_ && !prefill_,
                 "changing debug prefill chunk requires a positive size and idle deferred prefill");
        const int previous = opt_.prefill_chunk;
        opt_.prefill_chunk = chunk;
        return previous;
    }
    void debug_trim_expert_cache(int64_t bytes) { trim_cache(bytes); } // normal quiescent trim; graphs must stay valid
    void debug_reserve_kv(int tokens) { ensure_kv_capacity(tokens); } // normal checked growth; no positions are advanced
    std::vector<uint8_t> debug_prefix_state(); // canonical recurrent/pending-hidden/valid-KV bytes, independent of capacity
    std::vector<float> debug_hidden();   // copy of the current residual stream x (fp32)
    std::vector<float> logits_host();    // full logits of the last fed token
    /// A verification step over T tokens that keeps the first `keep` (MTP on); returns the T rows of logits.
    std::vector<float> debug_verify(const int* tokens, int T, int keep);
    std::vector<std::vector<float>>* debug_layers = nullptr;   // --no-graph only: token 0's x after every layer

private:
    friend class Prefill;
    void alloc_state();
    void setup_cache();
    void refill_cache_after_prefill(const std::vector<std::vector<int>>& previous_layout);
    void setup_async_cache();                // optional pinned staging + per-layer upload completion events
    void setup_miss_stream();                // freeze initially nonresident profile leaders after cache allocation
    int prepare_miss_stream(int layer, int T, cudaStream_t s); // partition completed mailbox, queue bounded DMA
    void ensure_kv_capacity(int tokens);      // grow backing storage without lowering the configured context
    void rebalance_cache();                  // optional bounded capacity move at a quiescent adapt boundary
    void trim_cache(int64_t bytes);   // drop the coldest cached experts until at least `bytes` are freed
    void check_decode_launch(cudaStream_t s, int il, const char* stage);
    void build_layer(cudaStream_t s, int il, int T, const StepParams* P, const float* next_w, float* snap);
    void build_decode(cudaStream_t s, int T);   // enqueue one decode step of T tokens (graph body)
    void build_mtp(cudaStream_t s, int T);      // enqueue one MTP step over T pending pairs (graph body)
    /// Runs a captured step (or builds it directly without graphs) with the CPU experts of layers l0..l1-1.
    void run_step(bool mtp, int T);
    void capture_event_graph(int T);          // CPU host-node branch, trunk only; never executes during capture
    void invalidate_event_graphs();           // caller has drained graph work before replacing backing pointers
    void capture_prompt_route_graph(int layer); // GPU-only T4 prefix, ending before cache readiness/router/mailbox
    void invalidate_prompt_route_graphs();     // caller has drained the compute stream before replacing KV
    void capture_prompt_full_graph(bool skip_head); // accepted T4 only; independent intermediate/final head variants
    void invalidate_prompt_full_graphs();      // caller drained compute/callback branches before KV replacement
    static void CUDART_CB event_graph_cpu(void* data); // CPU-only callback: no CUDA calls, including on workers
    void run_cpu_lm_head(bool mtp, int T);  // after the GPU trunk/MTP and CPU experts have finished
    void decode_one(int token);
    void decode_multi(const int* tokens, int T);   // trunk step over T tokens (verification)
    /// MTP over pending pairs. State-only callers do not need the discarded draft head (returns -1).
    int mtp_catch_up(int next, bool need_draft = true, bool defer_completion = false);
    int mtp_recur(int prev_draft, int pos);        // one more draft from the MTP's own hidden state
    /// After a verification step of T tokens: keep the first a (roll the delta net back), make them history.
    void commit_step(const int* toks, int a, int T);
    void adapt_cache();
    void apply_swap(int layer, int slot, int expert);
    /// Counts a sampled token for the penalties (and drops the one that leaves the window); stream-ordered.
    void count_sampled(int tok);

    EngineOptions opt_;
    int64_t cache_budget_extra_mib_ = 0;  // explicit opt-in beyond CUDA-reported free memory; not physical free VRAM
    Model model_;
    cudaStream_t stream_ = nullptr, copy_stream_ = nullptr;
    cudaGraphExec_t graph_[kMaxT + 1] = {};       // trunk step of T tokens
    cudaGraphExec_t mtp_graph_[kMaxT + 1] = {};   // MTP step over T pairs
    cudaGraphExec_t event_graph_[kMaxT + 1] = {}; // opt-in event-MoE trunk; distinct from legacy mapped-spin graphs
    struct EventGraphContext { Engine* engine; int layer, tokens; bool prompt = false; };
    std::vector<EventGraphContext> event_graph_contexts_; // allocated once; callback addresses never move
    cudaStream_t event_graph_cpu_stream_ = nullptr;
    cudaEvent_t event_graph_fork_ = nullptr, event_graph_join_ = nullptr;
    cudaEvent_t event_graph_complete_ = nullptr;
    bool event_graph_moe_ = false, event_graph_ready_ = false, capturing_event_graph_ = false;
    bool event_graph_blocking_ = false;
    uint64_t event_graph_steps_ = 0;
    uint64_t event_graph_captures_ = 0; // lifetime count, including recaptures after KV growth; debug regression evidence
    std::unique_ptr<CpuMoe> cpu_;
    std::unique_ptr<Prefill> prefill_;
    bool prefill_phase_switch_ = false;       // long prompts temporarily borrow the expert-cache VRAM
    bool prefill_snapshot_loan_ = false;      // long prompts also release unused MTP rollback frames
    std::vector<int8_t> head_q_;
    std::vector<float> head_d_, head_logits_;  // staging buffers for the optional CPU LM head
    bool head_last_only_ = false;             // feed's prompt batches need only the last row; never MTP verify
    bool prompt_skip_head_ = false, prompt_no_snapshots_ = false; // opt-in prompt-only work elision
    bool skip_trunk_head_ = false, skip_prompt_snapshots_ = false; // scoped to an accepted feed chunk
    uint64_t prompt_heads_skipped_ = 0, prompt_snapshot_bytes_skipped_ = 0;
    bool prompt_async_mtp_ = false;           // prompt-only KV catch-up shares the following trunk's completion wait
    uint64_t prompt_mtp_submitted_ = 0;       // submitted deferred catch-ups; GPU time is included in the next trunk
    bool prompt_route_graphs_ = false;       // bounded T4 prompt GPU prefixes; ordinary CPU/MoE scheduling follows
    std::vector<cudaGraphExec_t> prompt_route_graph_;
    cudaStream_t prompt_route_capture_stream_ = nullptr; // records graphs only; never executes model work
    uint64_t prompt_route_captures_ = 0, prompt_route_launches_ = 0;
    bool prompt_full_graph_ = false, capturing_prompt_full_graph_ = false;
    cudaGraphExec_t prompt_full_graph_exec_[2] = {}; // [0] final last-row head, [1] intermediate without head
    std::vector<EventGraphContext> prompt_full_contexts_; // fixed once at init; shared safely by serial graph launches
    cudaEvent_t prompt_full_complete_ = nullptr; // always blocking: main thread sleeps while callbacks/workers run
    uint64_t prompt_full_captures_ = 0, prompt_full_launches_[2] = {};
    bool sync_moe_ = false;                   // host-driven single-token handoff for older/WDDM GPUs
    bool event_moe_ = false;                  // host CUDA event + worker completion; no mapped-memory GPU wait
    bool event_blocking_ = false;             // opt in to OS-blocking router waits; default uses CUDA spin waits
    bool early_gpu_moe_ = false;              // opt in to enqueue GPU experts before waiting for the router mailbox
    int prompt_early_gpu_ = 2;               // missing=inherit global setting; explicit 0/1 affects prompt batches only
    bool async_cache_ = false;                // pinned expert uploads on copy_stream_, ordered per layer
    bool direct_pinned_cache_ = false;        // opt in to DMA from immutable, explicitly pinned model expert blobs
    bool trace_steps_ = false;
    Result* sync_res_dev_ = nullptr;          // copied CPU result; avoids device polling of host memory
    Mailbox* router_mb_dev_ = nullptr;        // event mode: GPU router writes device memory, then copies to host
    cudaEvent_t router_ready_ = nullptr;
    int miss_stream_eligible_limit_ = 0;     // STRATA_MISS_STREAM_EXPERTS, 0..256 fixed IDs/layer; default disabled
    int miss_stream_capacity_ = 0;           // staging slots = min(eligible limit, kMaxU); all routed eligible IDs fit
    size_t miss_stream_stride_ = 0;          // maximum original expert blob, budgeted before resident cache
    std::vector<std::vector<int>> miss_stream_eligible_; // fixed except explicit drained test override
    cudaStream_t miss_copy_stream_ = nullptr;
    cudaEvent_t miss_stream_ready_ = nullptr;
    uint8_t* miss_stream_bank_ = nullptr;
    HitList* miss_stream_hits_host_ = nullptr; // pinned, reused only after the next router completes
    HitList* miss_stream_hits_dev_ = nullptr;
    float* miss_stream_out_ = nullptr;
    bool miss_stream_serialized_ = false;    // debug reference, identical arithmetic and placement
    int miss_stream_last_layer_ = -1;
    std::vector<int> miss_stream_last_ids_;

    // host-mapped handshake memory
    Mailbox* mb_host_ = nullptr;
    Mailbox* mb_dev_ = nullptr;
    Result* res_host_ = nullptr;
    Result* res_dev_ = nullptr;
    StepParams* params_host_ = nullptr;       // pinned; copied into params_dev_ by each step
    StepParams* params_dev_ = nullptr;
    StepParams* mparams_host_ = nullptr;      // same for MTP steps
    StepParams* mparams_dev_ = nullptr;
    StepParams* mtp_tail_host_ = nullptr;     // separate immutable-in-flight header; shifted to the last pending pair
    StepParams* mtp_tail_dev_ = nullptr;
    int32_t* out_host_ = nullptr;             // mapped: [0..kMaxT) verification argmax, [kMaxT] MTP draft
    int32_t* out_dev_ = nullptr;
    float* prob_host_ = nullptr;              // mapped: the MTP's probability of its draft
    float* prob_dev_ = nullptr;
    unsigned long long* ecount_ = nullptr;    // device: resident / missing (expert, token) routes
    Candidates* cand_host_ = nullptr;
    Candidates* cand_dev_ = nullptr;
    int32_t* tok_host_ = nullptr;   // mapped
    int32_t* tok_dev_ = nullptr;

    // device buffers (decode)
    float *x_ = nullptr, *xn_ = nullptr, *qkvz_ = nullptr, *ba_ = nullptr, *conv_out_ = nullptr, *o_ = nullptr;
    float* gdn_prompt_factors_ = nullptr;      // optional per-head/token factors, reused in prompt batches only
    float *qkv_ = nullptr, *qbuf_ = nullptr, *part_o_ = nullptr, *part_ml_ = nullptr, *attn_out_ = nullptr;
    float *rlog_ = nullptr, *shgu_ = nullptr, *shout_ = nullptr, *h_gu_ = nullptr, *moe_gpu_ = nullptr;
    float *logits_ = nullptr, *d_sg_ = nullptr, *cand_tmp_ = nullptr;
    // MTP
    bool mtp_on_ = false;
    bool mtp_state_only_ = false;              // scoped catch-up mode: retain MTP state, skip its discarded head
    bool mtp_last_row_ = false;                // single full-attention MTP: earlier pending pairs only require KV
    uint64_t mtp_last_row_steps_ = 0, mtp_kv_only_steps_ = 0;
    int draft_ = 0;
    float *mtp_hin_ = nullptr, *mtp_hlast_ = nullptr, *mtp_logits_ = nullptr;
    std::unique_ptr<MtpDraftSubset> mtp_subset_; // optional dynamic draft head; target logits remain full-vocabulary
    ActQ mtp_act_;                            // [enorm(e) | hnorm(h)] per pair, 4096 each
    int mtp_k_ = 0;                           // pending pairs: positions n_past_-mtp_k_ .. n_past_-1, hiddens in mtp_hin_
    float* snap_arena_ = nullptr;             // delta-net state after each token of a verification step (but the last)
    int32_t *d_nmiss_ = nullptr, *d_err_ = nullptr;
    unsigned long long* wait_ns_ = nullptr;     // GPU time spent waiting for the CPU experts (decode)
    HitList* hits_ = nullptr;
    ActQ xq_, yq_, shq_, hq_;
    uint32_t* route_counts_ = nullptr;   // [n_layer][256] (device, cumulative)
    uint32_t* tok_counts_ = nullptr;     // [vocab] token counts of the penalty window
    std::vector<int> pen_hist_;          // the tokens counted, oldest first (the window is the last pen_n_)
    int pen_n_ = 0;

    // state
    std::vector<uint16_t*> kc_, vc_;          // per layer (null for delta-net layers)
    bool kv_int8_ = false;                  // explicit approximate KV opt-in; FP16 is the default
    int kv_initial_tokens_ = 0;             // 0 reserves the full configured context as before
    int kv_capacity_ = 0;                   // allocated head stride, independent of opt_.ctx
    std::vector<int8_t*> kc_i8_, vc_i8_;
    std::vector<float*> kc_scale_, vc_scale_;
    std::vector<float*> conv_st_, ssm_st_;    // per layer (null for attention layers)
    float* state_arena_ = nullptr;            // conv + ssm states of all delta-net layers, contiguous
    struct Checkpoint {
        float* arena = nullptr;               // copy of state_arena_
        float* hin = nullptr;                 // copy of mtp_hin_ (MTP pending pairs)
        int mtp_k = 0;
        std::vector<int> tokens;              // the tokens it covers
        uint64_t stamp = 0;                   // LRU
    };
    std::vector<Checkpoint> ckpts_;
    uint64_t ckpt_clock_ = 0;
    int64_t state_floats_ = 0;
    int n_past_ = 0;
    std::vector<int> history_;
    bool logits_valid_ = false;
    uint32_t seq_ = 0;

    // expert cache
    std::vector<uint8_t*> slot_base_;         // per layer
    std::vector<int> slots_per_layer_;
    std::vector<std::vector<int>> slot_expert_;   // [layer][slot] -> expert
    std::vector<int32_t> residency_host_;     // [layer][256] -> slot or -1
    int32_t* residency_dev_ = nullptr;
    CacheLayerInfo* cinfo_dev_ = nullptr;
    int64_t cache_slots_total_ = 0;
    int64_t cache_bytes_ = 0;
    std::vector<double> freq_;                // [layer][256] routing frequency (profile + decayed live counts)
    std::vector<uint32_t> counts_seen_;       // last copy of route_counts_
    std::vector<float> prior_;                // the profile as loaded (saved back plus this session's counts)
    int steps_since_adapt_ = 0;
    std::vector<uint32_t> route_host_;
    uint8_t* cache_upload_host_ = nullptr;    // pinned, immutable until cache_upload_done_ completes
    size_t cache_upload_capacity_ = 0;
    int32_t* cache_residency_host_ = nullptr; // pinned snapshot, same lifetime as the blob staging buffer
    std::vector<cudaEvent_t> cache_layer_ready_;
    cudaEvent_t cache_upload_done_ = nullptr;
    cudaEvent_t cache_upload_started_ = nullptr;
    size_t cache_upload_bytes_ = 0;
    uint64_t direct_pinned_uploads_ = 0;       // actual async adaptive uploads; separate from logical swap count
    double cache_upload_bytes_per_ms_ = 0;
    int rebalance_every_ = 0;                 // routed trunk tokens; 0 preserves fixed layer capacities
    int64_t rebalance_max_bytes_ = 0;
    uint64_t rebalance_tokens_ = 0;
    std::vector<double> cpu_pair_ms_;         // smoothed CPU job time per routed expert/token pair, no LM head
    double rebalance_cost_per_byte_ = 0;      // measured allocation + copy cost from earlier rebalances
    // Diagnostic only: CUDA event spans include stream idle gaps, not just kernel execution.
    // Disabled for throughput trials; they distinguish the route/head path from CPU/GPU MoE overlap.
    struct PipelineLayer {
        cudaEvent_t event[5] = {};
        double cpu_ms = 0, cpu_wait_ms = 0, router_wait_ms = 0;
    };
    bool profile_pipeline_ = false;
    bool flush_moe_submit_ = false;          // opt-in WDDM nonblocking queue submission before CPU drain
    std::vector<PipelineLayer> pipeline_layers_;
    cudaEvent_t pipeline_end_ = nullptr;
};

}  // namespace sq
