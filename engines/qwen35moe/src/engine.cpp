// engine.cpp - see engine.hpp.
#include "engine.hpp"

#include "common.hpp"
#include "cpu_lm_head.hpp"
#include "mtp_subset.cuh"
#include "prefill.hpp"
#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <thread>

namespace sq {

namespace {

int64_t env_mib(const char* name, int64_t fallback, int64_t maximum) {
    const char* value = std::getenv(name);
    if (!value) return fallback;
    SQ_CHECK(*value != '\0', "%s must be an integer from 0 to %lld", name, (long long)maximum);
    int64_t result = 0;
    for (const char* p = value; *p; ++p) {
        SQ_CHECK(*p >= '0' && *p <= '9', "%s must be an integer from 0 to %lld", name, (long long)maximum);
        result = result * 10 + (*p - '0');
        SQ_CHECK(result <= maximum, "%s must be an integer from 0 to %lld", name, (long long)maximum);
    }
    return result;
}

template <typename T>
T* dmalloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    // The engine uses nonblocking streams: they do not wait for this default-stream initialization.
    // Complete it before returning, or a later upload (notably residency=-1) can be overwritten by zero.
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    return p;
}

ActQ alloc_act(int n) {
    ActQ a;
    a.q = dmalloc<int8_t>(n);
    a.d = dmalloc<float>(n / 32);
    a.s = dmalloc<float>(n / 32);
    return a;
}

template <typename T>
void host_mapped(size_t n, T*& host, T*& dev) {
    CUDA_CHECK(cudaHostAlloc((void**)&host, n * sizeof(T), cudaHostAllocMapped));
    std::memset((void*)host, 0, n * sizeof(T));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&dev, (void*)host, 0));
}

constexpr uint32_t kProfileMagic = 0x50455153;   // "SQEP"

}  // namespace

Engine::Engine() = default;

Engine::~Engine() {
    // Graph host callbacks may still need the CPU pool. Drain every graph branch before stopping its workers.
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    if (miss_copy_stream_) cudaStreamSynchronize(miss_copy_stream_);
    if (stream_) cudaStreamSynchronize(stream_);
    if (event_graph_cpu_stream_) cudaStreamSynchronize(event_graph_cpu_stream_);
    if (cpu_) cpu_->stop();
    invalidate_event_graphs();
    invalidate_prompt_route_graphs();
    invalidate_prompt_full_graphs();
    if (prompt_route_capture_stream_) cudaStreamDestroy(prompt_route_capture_stream_);
    prefill_.reset();
    for (auto* g : graph_)
        if (g) cudaGraphExecDestroy(g);
    for (auto* g : mtp_graph_)
        if (g) cudaGraphExecDestroy(g);
    for (auto e : cache_layer_ready_)
        if (e) cudaEventDestroy(e);
    if (cache_upload_done_) cudaEventDestroy(cache_upload_done_);
    if (cache_upload_started_) cudaEventDestroy(cache_upload_started_);
    if (router_ready_) cudaEventDestroy(router_ready_);
    if (miss_stream_ready_) cudaEventDestroy(miss_stream_ready_);
    if (miss_copy_stream_) cudaStreamDestroy(miss_copy_stream_);
    if (miss_stream_hits_host_) cudaFreeHost(miss_stream_hits_host_);
    if (miss_stream_hits_dev_) cudaFree(miss_stream_hits_dev_);
    if (miss_stream_bank_) cudaFree(miss_stream_bank_);
    if (miss_stream_out_) cudaFree(miss_stream_out_);
    for (auto& layer : pipeline_layers_)
        for (auto event : layer.event) if (event) cudaEventDestroy(event);
    if (pipeline_end_) cudaEventDestroy(pipeline_end_);
    if (cache_upload_host_) cudaFreeHost(cache_upload_host_);
    if (cache_residency_host_) cudaFreeHost(cache_residency_host_);
    if (router_mb_dev_) cudaFree(router_mb_dev_);
    if (mtp_tail_host_) cudaFreeHost(mtp_tail_host_);
    if (mtp_tail_dev_) cudaFree(mtp_tail_dev_);
    if (event_graph_fork_) cudaEventDestroy(event_graph_fork_);
    if (event_graph_join_) cudaEventDestroy(event_graph_join_);
    if (event_graph_complete_) cudaEventDestroy(event_graph_complete_);
    if (prompt_full_complete_) cudaEventDestroy(prompt_full_complete_);
    if (event_graph_cpu_stream_) cudaStreamDestroy(event_graph_cpu_stream_);
    if (gdn_prompt_factors_) cudaFree(gdn_prompt_factors_);
}

bool Engine::init(const EngineOptions& opt, std::string& err) {
    opt_ = opt;
    prefill_phase_switch_ = env_mib("STRATA_PREFILL_PHASE_SWITCH", 0, 1) != 0;
    prefill_snapshot_loan_ = env_mib("STRATA_PREFILL_SNAPSHOT_LOAN", 0, 1) != 0;
    SQ_CHECK(!prefill_snapshot_loan_ || prefill_phase_switch_,
             "STRATA_PREFILL_SNAPSHOT_LOAN requires STRATA_PREFILL_PHASE_SWITCH=1");
    if (env_mib("STRATA_NATIVE_DENSE_Q6", 0, 1) != 0) {
        SQ_CHECK(opt_.prefill_chunk == 0,
                 "STRATA_NATIVE_DENSE_Q6 requires --prefill-chunk 0; large prefill assumes expanded Q8 weights");
        const char* cpu_head = std::getenv("STRATA_CPU_LM_HEAD");
        SQ_CHECK(!cpu_head || std::string(cpu_head) != "1",
                 "STRATA_NATIVE_DENSE_Q6 currently requires the GPU LM head");
    }
    kv_int8_ = env_mib("STRATA_KV_INT8", 0, 1) != 0;
    kv_initial_tokens_ = (int)env_mib("STRATA_KV_INITIAL_TOKENS", 0, 1048576);
    if (kv_int8_) {
        opt_.prefill_chunk = 0; // the large prefill path still writes FP16 KV
        log("INT8 KV enabled (approximate, per-head FP32 scales); bounded decode/microbatch path only");
    }
    if (kv_initial_tokens_ > 0) {
        opt_.use_graph = false;     // growth changes KV pointers and per-head strides
        if (!prefill_phase_switch_) opt_.prefill_chunk = 0;
    }
    SQ_CHECK(opt_.vram_reserve_mb >= 0, "runtime VRAM reserve must be nonnegative");
    SQ_CHECK(std::isfinite(opt_.adapt_decay) && opt_.adapt_decay >= 0.0 && opt_.adapt_decay < 1.0,
             "--adapt-decay must be finite and satisfy 0 <= decay < 1");
    cache_budget_extra_mib_ = env_mib("STRATA_CACHE_BUDGET_EXTRA_MIB", 0, 1024);
    const char* sync_env = std::getenv("STRATA_SYNC_MOE");
    sync_moe_ = sync_env && std::string(sync_env) == "1";
    event_moe_ = env_mib("STRATA_EVENT_MOE", 0, 1) != 0;
    miss_stream_eligible_limit_ = (int)env_mib("STRATA_MISS_STREAM_EXPERTS", 0, 256);
    miss_stream_capacity_ = std::min(miss_stream_eligible_limit_, kMaxU);
    mtp_last_row_ = env_mib("STRATA_MTP_LAST_ROW", 0, 1) != 0;
    event_blocking_ = env_mib("STRATA_EVENT_BLOCKING", 0, 1) != 0;
    early_gpu_moe_ = env_mib("STRATA_MOE_EARLY_GPU", 0, 1) != 0;
    // env_mib returns the fallback only when absent; supplied values remain strictly bounded to 0/1.
    prompt_early_gpu_ = (int)env_mib("STRATA_PROMPT_EARLY_GPU", 2, 1);
    event_graph_moe_ = env_mib("STRATA_EVENT_MOE_GRAPH", 0, 1) != 0;
    prompt_skip_head_ = env_mib("STRATA_PROMPT_SKIP_HEAD", 0, 1) != 0;
    prompt_no_snapshots_ = env_mib("STRATA_PROMPT_NO_SNAPSHOTS", 0, 1) != 0;
    prompt_async_mtp_ = env_mib("STRATA_PROMPT_ASYNC_MTP", 0, 1) != 0;
    prompt_route_graphs_ = env_mib("STRATA_PROMPT_ROUTE_GRAPH", 0, 1) != 0;
    prompt_full_graph_ = env_mib("STRATA_PROMPT_FULL_GRAPH", 0, 1) != 0;
    event_graph_blocking_ = env_mib("STRATA_EVENT_GRAPH_BLOCKING", 0, 1) != 0;
    profile_pipeline_ = env_mib("STRATA_PROFILE_PIPELINE", 0, 1) != 0;
    flush_moe_submit_ = env_mib("STRATA_MOE_FLUSH", 0, 1) != 0;
    if (profile_pipeline_) opt_.use_graph = false;
    async_cache_ = env_mib("STRATA_ASYNC_CACHE", 0, 1) != 0;
    direct_pinned_cache_ = env_mib("STRATA_CACHE_DIRECT_PINNED", 0, 1) != 0;
    rebalance_every_ = (int)env_mib("STRATA_CACHE_REBALANCE_EVERY", 0, 65536);
    rebalance_max_bytes_ = env_mib("STRATA_CACHE_REBALANCE_MAX_MIB", 128, 512) * 1048576ll;
    SQ_CHECK(!(sync_moe_ && event_moe_), "STRATA_EVENT_MOE=1 requires STRATA_SYNC_MOE=0");
    SQ_CHECK(!(prompt_skip_head_ || prompt_no_snapshots_) || (event_moe_ && !event_graph_moe_),
             "prompt work elision requires STRATA_EVENT_MOE=1 and STRATA_EVENT_MOE_GRAPH=0");
    SQ_CHECK(!prompt_async_mtp_ || (event_moe_ && mtp_last_row_ && !event_graph_moe_ && !profile_pipeline_),
             "STRATA_PROMPT_ASYNC_MTP requires EVENT_MOE=1, MTP_LAST_ROW=1, EVENT_MOE_GRAPH=0 and PROFILE_PIPELINE=0");
    SQ_CHECK(!prompt_route_graphs_ || (event_moe_ && prompt_no_snapshots_ && !event_graph_moe_ && !profile_pipeline_),
             "STRATA_PROMPT_ROUTE_GRAPH requires EVENT_MOE=1, PROMPT_NO_SNAPSHOTS=1, EVENT_MOE_GRAPH=0 and PROFILE_PIPELINE=0");
    SQ_CHECK(!prompt_full_graph_ || (event_moe_ && prompt_no_snapshots_ && !event_graph_moe_ &&
                 !prompt_route_graphs_ && !profile_pipeline_ && !miss_stream_capacity_),
             "STRATA_PROMPT_FULL_GRAPH requires EVENT_MOE=1, PROMPT_NO_SNAPSHOTS=1, EVENT_MOE_GRAPH=0, "
             "PROMPT_ROUTE_GRAPH=0, PROFILE_PIPELINE=0 and MISS_STREAM_EXPERTS=0");
    SQ_CHECK(!mtp_last_row_ || event_moe_, "STRATA_MTP_LAST_ROW=1 requires STRATA_EVENT_MOE=1");
    SQ_CHECK(!early_gpu_moe_ || event_moe_, "STRATA_MOE_EARLY_GPU=1 requires STRATA_EVENT_MOE=1");
    SQ_CHECK(prompt_early_gpu_ == 2 || event_moe_, "STRATA_PROMPT_EARLY_GPU requires STRATA_EVENT_MOE=1");
    if (prompt_early_gpu_ != 2)
        log("prompt expert scheduling override: early GPU submission %d for last-head prompt batches", prompt_early_gpu_);
    SQ_CHECK(!event_graph_moe_ || event_moe_, "STRATA_EVENT_MOE_GRAPH=1 requires STRATA_EVENT_MOE=1");
    SQ_CHECK(!event_graph_blocking_ || event_graph_moe_,
             "STRATA_EVENT_GRAPH_BLOCKING=1 requires STRATA_EVENT_MOE_GRAPH=1");
    SQ_CHECK(!event_graph_moe_ || !profile_pipeline_, "STRATA_EVENT_MOE_GRAPH requires STRATA_PROFILE_PIPELINE=0");
    SQ_CHECK(!miss_stream_capacity_ || (event_moe_ && !event_graph_moe_ && !opt.use_graph),
             "STRATA_MISS_STREAM_EXPERTS requires event MoE, --no-graph, and STRATA_EVENT_MOE_GRAPH=0");
    SQ_CHECK(!async_cache_ || event_moe_ || sync_moe_,
             "STRATA_ASYNC_CACHE=1 requires STRATA_EVENT_MOE=1 or STRATA_SYNC_MOE=1 (no GPU/host spin handoff)");
    SQ_CHECK(!direct_pinned_cache_ || async_cache_, "STRATA_CACHE_DIRECT_PINNED=1 requires STRATA_ASYNC_CACHE=1");
    SQ_CHECK(rebalance_every_ == 0 || (event_moe_ && async_cache_ && opt_.adapt_every > 0),
             "STRATA_CACHE_REBALANCE_EVERY requires event MoE, asynchronous cache, and --adapt-every > 0");
    const char* trace_env = std::getenv("STRATA_TRACE_STEPS");
    trace_steps_ = trace_env && std::string(trace_env) == "1";
    SQ_CHECK(!event_graph_moe_ || !trace_steps_, "STRATA_EVENT_MOE_GRAPH requires STRATA_TRACE_STEPS=0");
    SQ_CHECK(!prompt_route_graphs_ || !trace_steps_, "STRATA_PROMPT_ROUTE_GRAPH requires STRATA_TRACE_STEPS=0");
    SQ_CHECK(!prompt_full_graph_ || !trace_steps_, "STRATA_PROMPT_FULL_GRAPH requires STRATA_TRACE_STEPS=0");
    if (sync_moe_) {
        opt_.mtp_draft = 0;
        opt_.prefill_chunk = 0;
        opt_.use_graph = false;
        log("synchronous CPU MoE enabled: single-token decode, MTP and CUDA graphs disabled");
    }
    if (event_moe_ || async_cache_) {
        // Host event waits cannot be captured.  Batched prefill has its own routing path and does not consume
        // these per-layer cache readiness events, so keep both opt-ins on the audited decode path.
        opt_.use_graph = false;
        if (!prefill_phase_switch_) opt_.prefill_chunk = 0;
        if (event_moe_)
            log("event CPU MoE enabled: %s router wait, CPU/GPU expert overlap, 1..%d tokens per step",
                event_blocking_ ? "blocking" : "spin", kMaxT);
        if (early_gpu_moe_)
            log("event MoE early GPU submission: enqueue shared/resident experts before the host router wait");
        if (async_cache_)
            log("asynchronous expert cache uploads enabled: pinned staging and per-layer CUDA events");
        if (direct_pinned_cache_)
            log("adaptive expert uploads use immutable pinned model blobs directly; pageable sources retain staging");
        if (rebalance_every_ > 0)
            log("bounded global expert cache rebalance: every %d routed trunk tokens, at most %.0f MiB copied per move",
                rebalance_every_, rebalance_max_bytes_ / 1048576.0);
    }
    opt_.mtp_draft = std::clamp(opt_.mtp_draft, 0, kMaxT - 1);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    log("GPU: %s, %.1f GiB, sm_%d%d", prop.name, prop.totalGlobalMem / 1073741824.0, prop.major, prop.minor);
    if (!model_.load(opt.model_path, opt_.mtp_draft > 0, err)) return false;
    const Config& c = model_.cfg;
    if (model_.cpu_lm_head) {
        opt_.use_graph = false;
        opt_.prefill_chunk = 0;
        head_q_.resize((size_t)kMaxT * c.n_embd);
        head_d_.resize((size_t)kMaxT * c.n_embd / 32);
        head_logits_.resize((size_t)kMaxT * c.n_vocab);
        log("CPU LM head enabled: decode-only prompts and uncaptured steps");
    }
    SQ_CHECK(!prompt_full_graph_ || (!model_.cpu_lm_head && !opt_.use_graph),
             "STRATA_PROMPT_FULL_GRAPH requires a GPU LM head and uncaptured ordinary decode/MTP steps");
    mtp_on_ = c.n_mtp > 0 && opt_.mtp_draft > 0;
    if (mtp_last_row_) {
        SQ_CHECK(mtp_on_ && c.n_mtp == 1 && model_.layers[c.n_layer].attn && !opt_.use_graph,
                 "STRATA_MTP_LAST_ROW requires one full-attention MTP layer and uncaptured MTP steps");
        log("MTP last-row mode: update all pending KV, execute only the final pair's attention/MoE; state-only catch-up stops at KV");
        log("MTP routing/profile counters count actual executed experts; discarded pending-row MoE is omitted");
    }
    if (opt_.mtp_draft > 0 && !mtp_on_) log("no MTP block in this model: speculative decoding off");
    SQ_CHECK(!prompt_async_mtp_ || (mtp_on_ && !opt_.use_graph),
             "STRATA_PROMPT_ASYNC_MTP requires active MTP and uncaptured steps");
    if (prompt_async_mtp_)
        log("prompt MTP KV catch-up shares the next trunk wait; deferred GPU time is in feed/trunk time, not mtp_ms (submission only)");
    opt_.ctx = std::min(opt_.ctx, c.ctx_train);
    opt_.ctx = (opt_.ctx + 255) / 256 * 256;
    if (prefill_phase_switch_)
        opt_.prefill_min = std::max(opt_.prefill_min,
            (int)env_mib("STRATA_PREFILL_PHASE_MIN", 512, opt_.ctx));
    kv_capacity_ = opt_.ctx;
    if (kv_initial_tokens_ > 0) {
        const int wanted = std::min(kv_initial_tokens_, opt_.ctx);
        kv_capacity_ = std::min(256, opt_.ctx);
        while (kv_capacity_ < wanted)
            kv_capacity_ = kv_capacity_ > opt_.ctx / 2 ? opt_.ctx : kv_capacity_ * 2;
        log("lazy KV enabled: initial capacity %d, configured context %d; grows on demand by releasing expert slots",
            kv_capacity_, opt_.ctx);
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    if (profile_pipeline_) {
        SQ_CHECK(event_moe_, "STRATA_PROFILE_PIPELINE requires event MoE");
        pipeline_layers_.resize(c.n_moe_layers());
        for (auto& layer : pipeline_layers_)
            for (auto& event : layer.event) CUDA_CHECK(cudaEventCreate(&event));
        CUDA_CHECK(cudaEventCreate(&pipeline_end_));
        log("pipeline diagnostic enabled: event spans include stream/host idle gaps; disable for final throughput");
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    if (prompt_route_graphs_) {
        SQ_CHECK(!opt_.use_graph, "prompt route graphs require uncaptured ordinary decode/MTP steps");
        prompt_route_graph_.resize(c.n_layer, nullptr);
        CUDA_CHECK(cudaStreamCreateWithFlags(&prompt_route_capture_stream_, cudaStreamNonBlocking));
        log("prompt GPU route graphs enabled: T4 accepted chunks only, no snapshots; ordinary router/CPU/cache scheduling");
        log("route graphs instantiate lazily and report CUDA free memory; no extra cache capacity is claimed or silently trimmed");
    }
    if (event_graph_moe_ || prompt_full_graph_) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&event_graph_cpu_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&event_graph_fork_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&event_graph_join_, cudaEventDisableTiming));
    }
    if (event_graph_moe_) {
        if (event_graph_blocking_)
            CUDA_CHECK(cudaEventCreateWithFlags(&event_graph_complete_, cudaEventDisableTiming | cudaEventBlockingSync));
        event_graph_contexts_.resize((size_t)(kMaxT + 1) * c.n_layer);
        for (int t = 1; t <= kMaxT; ++t)
            for (int layer = 0; layer < c.n_layer; ++layer)
                event_graph_contexts_[(size_t)t * c.n_layer + layer] = EventGraphContext{this, layer, t};
        log("event MoE graphs enabled independently of --no-graph: trunk only; MTP and prompt last-head stay uncaptured");
        log("event MoE graphs wait for all pending expert uploads before launch; lazy KV growth invalidates graphs");
        if (event_graph_blocking_)
            log("event MoE graph completion uses one blocking host wait per trunk window");
    }
    if (prompt_full_graph_) {
        CUDA_CHECK(cudaEventCreateWithFlags(&prompt_full_complete_, cudaEventDisableTiming | cudaEventBlockingSync));
        prompt_full_contexts_.resize(c.n_layer);
        for (int layer = 0; layer < c.n_layer; ++layer)
            prompt_full_contexts_[layer] = EventGraphContext{this, layer, 4, true};
        log("prompt full graphs enabled: accepted T4 only, separate intermediate/final head graphs, no snapshots");
        log("prompt graph callbacks use prompt CPU completion; one blocking host wait per window; MTP/tails stay uncaptured");
        log("prompt graphs preserve adaptation cadence, wait for queued cache uploads, and invalidate on lazy KV growth");
    }
    alloc_state();
    if (opt_.prefill_chunk > 0 && !prefill_phase_switch_) {
        prefill_ = std::make_unique<Prefill>(*this);
        if (!prefill_->init(err)) return false;
    } else if (opt_.prefill_chunk > 0) {
        log("phase-switched prefill enabled: long prompts borrow expert-cache VRAM, then refill from prompt routes");
    } else {
        log("batched prefill disabled: prompts use the decode path without prefill workspaces");
    }
    setup_cache();
    if (miss_stream_capacity_) setup_miss_stream();
    if (async_cache_) setup_async_cache();

    const int NL = c.n_moe_layers();
    std::vector<ExpertLayerDesc> descs(NL);
    for (int il = 0; il < NL; ++il) {
        const LayerW& L = model_.layers[il];
        descs[il] = ExpertLayerDesc{L.experts, L.blob_bytes, L.gu_bytes, L.t_gu, L.t_down, c.n_expert};
    }
    cpu_ = std::make_unique<CpuMoe>();
    cpu_->start(opt_.cpu_threads, NL, descs, mb_host_, res_host_, opt_.pin_threads);

    {   // Load the active kernels now instead of on the first request.  The decode-only path needs no cuBLAS.
        const int warm_tokens = prefill_ && prefill_->ready() ? std::max(opt_.prefill_min, 16) : (mtp_on_ ? 2 : 1);
        std::vector<int> w(warm_tokens, 198);
        const auto saved = opt_.adapt_every;
        opt_.adapt_every = 0;
        feed(w.data(), (int)w.size());
        opt_.adapt_every = saved;
        CUDA_CHECK(cudaMemsetAsync(route_counts_, 0, (size_t)NL * c.n_expert * 4, stream_));
        stats = EngineStats{};
        CUDA_CHECK(cudaMemsetAsync(wait_ns_, 0, 8, stream_));
        CUDA_CHECK(cudaMemsetAsync(ecount_, 0, 16, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    if (opt_.use_graph) {
        auto capture = [&](cudaGraphExec_t& ge, auto&& body) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            body();
            CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
            CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
            CUDA_CHECK(cudaGraphDestroy(g));
        };
        const int tmax = mtp_on_ ? opt_.mtp_draft + 1 : 1;
        for (int T = 1; T <= tmax; ++T) {
            capture(graph_[T], [&] { build_decode(stream_, T); });
            if (mtp_on_) capture(mtp_graph_[T], [&] { build_mtp(stream_, T); });
        }
    }
    if (event_graph_moe_) {
        const int tmax = mtp_on_ ? opt_.mtp_draft + 1 : 1;
        for (int t = 1; t <= tmax; ++t) capture_event_graph(t);
        event_graph_ready_ = true;
    }
    reset();
    size_t fr = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    // The warm-up and the graphs allocated more (cuBLAS workspace, lazily loaded kernels, graph memory) than
    // setup_cache could see: give experts back until the reserve is free.  The explicit extra-budget opt-in
    // uses the same effective free memory here as at allocation; actual CUDA free is still reported below.
    const int64_t effective_free = (int64_t)fr + cache_budget_extra_mib_ * 1048576ll;
    const int64_t short_by = opt_.vram_reserve_mb * 1048576ll - effective_free;
    if (short_by > 0) {
        trim_cache(short_by);
        CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    }
    if (cache_budget_extra_mib_ > 0)
        log("expert cache extra-budget opt-in: actual CUDA free %.1f MiB, extra allowance %lld MiB, "
            "effective free %.1f MiB, runtime reserve %lld MiB",
            fr / 1048576.0, (long long)cache_budget_extra_mib_, fr / 1048576.0 + cache_budget_extra_mib_,
            (long long)opt_.vram_reserve_mb);
    log("ready: ctx %d, expert cache %lld slots (%.2f GiB, %.1f%% of experts), %d CPU threads, VRAM free %.2f GiB",
        opt_.ctx, (long long)cache_slots_total_, cache_gib(), 100.0 * cache_slots_total_ / (NL * c.n_expert),
        opt_.cpu_threads, fr / 1073741824.0);
    draft_ = mtp_on_ ? opt_.mtp_draft : 0;
    if (mtp_on_) log("MTP speculative decoding: %d draft token(s) per step", opt_.mtp_draft);
    return true;
}

void Engine::alloc_state() {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers();
    constexpr int T = kMaxT;
    host_mapped(NL, mb_host_, mb_dev_);
    host_mapped(NL, res_host_, res_dev_);
    if (sync_moe_ || event_moe_) sync_res_dev_ = dmalloc<Result>(1);
    if (event_moe_) {
        router_mb_dev_ = dmalloc<Mailbox>(NL);
        CUDA_CHECK(cudaEventCreateWithFlags(&router_ready_, cudaEventDisableTiming |
                                           (event_blocking_ ? cudaEventBlockingSync : 0)));
        cpu_pair_ms_.assign(NL, 0.0);
    }
    if (miss_stream_capacity_) {
        for (const auto& layer : model_.layers)
            miss_stream_stride_ = std::max(miss_stream_stride_, (size_t)layer.blob_bytes);
        SQ_CHECK(miss_stream_stride_ > 0, "transient expert staging requires nonempty original blobs");
        miss_stream_bank_ = dmalloc<uint8_t>((size_t)miss_stream_capacity_ * miss_stream_stride_);
        miss_stream_hits_dev_ = dmalloc<HitList>(1);
        miss_stream_out_ = dmalloc<float>((size_t)kMaxT * c.n_embd);
        CUDA_CHECK(cudaHostAlloc((void**)&miss_stream_hits_host_, sizeof(HitList), cudaHostAllocDefault));
        std::memset(miss_stream_hits_host_, 0, sizeof(HitList));
        CUDA_CHECK(cudaStreamCreateWithFlags(&miss_copy_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&miss_stream_ready_, cudaEventDisableTiming));
        miss_stream_last_ids_.reserve(miss_stream_capacity_);
        log("experimental transient misses: up to %d fixed eligible experts/layer, %d staging slots, %.2f MiB staging before cache budget",
            miss_stream_eligible_limit_, miss_stream_capacity_, miss_stream_capacity_ * miss_stream_stride_ / 1048576.0);
        log("transient experts retain original weights; GPU activation rounding differs from CPU. Streamed misses are not cache hits");
    }
    host_mapped(1, cand_host_, cand_dev_);
    host_mapped(4, tok_host_, tok_dev_);
    host_mapped(kMaxT + 4, out_host_, out_dev_);
    host_mapped(4, prob_host_, prob_dev_);
    d_err_ = tok_dev_ + 1;   // tok_host_[1] is the GPU error flag
    wait_ns_ = dmalloc<unsigned long long>(1);
    ecount_ = dmalloc<unsigned long long>(2);
    // the params structs must live in device memory for the kernels: each step copies them in as its first node
    CUDA_CHECK(cudaHostAlloc((void**)&params_host_, sizeof(StepParams), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&mparams_host_, sizeof(StepParams), cudaHostAllocDefault));
    std::memset(params_host_, 0, sizeof(StepParams));
    std::memset(mparams_host_, 0, sizeof(StepParams));
    params_dev_ = dmalloc<StepParams>(1);
    mparams_dev_ = dmalloc<StepParams>(1);
    if (mtp_last_row_) {
        CUDA_CHECK(cudaHostAlloc((void**)&mtp_tail_host_, sizeof(StepParams), cudaHostAllocDefault));
        std::memset(mtp_tail_host_, 0, sizeof(StepParams));
        mtp_tail_dev_ = dmalloc<StepParams>(1);
    }

    x_ = dmalloc<float>((size_t)T * c.n_embd);
    xn_ = dmalloc<float>((size_t)T * c.n_embd);
    qkvz_ = dmalloc<float>((size_t)T * (c.conv_dim() + c.ssm_d_inner));
    ba_ = dmalloc<float>((size_t)T * 64);
    conv_out_ = dmalloc<float>((size_t)T * c.conv_dim());
    if (gdn_prepare_enabled()) {
        gdn_prompt_factors_ = dmalloc<float>((size_t)T * 32 * 4);
        log("GDN invariant factors prepared once per head/token in prompt microbatches only");
    }
    o_ = dmalloc<float>((size_t)T * c.ssm_d_inner);
    qkv_ = dmalloc<float>((size_t)T * (c.n_head * c.head_dim * 2 + 2 * c.n_head_kv * c.head_dim));
    qbuf_ = dmalloc<float>((size_t)T * c.n_head * c.head_dim);
    const int ms = attn_max_splits(opt_.ctx);
    part_o_ = dmalloc<float>((size_t)T * c.n_head * ms * c.head_dim);
    part_ml_ = dmalloc<float>((size_t)T * c.n_head * ms * 2);
    attn_out_ = dmalloc<float>((size_t)T * c.n_embd);
    rlog_ = dmalloc<float>((size_t)T * (c.n_expert + 1));
    shgu_ = dmalloc<float>((size_t)T * 2 * c.n_ff_shexp);
    shout_ = dmalloc<float>((size_t)T * c.n_embd);
    h_gu_ = dmalloc<float>((size_t)kMaxU * 2 * c.n_ff_exp);
    moe_gpu_ = dmalloc<float>((size_t)T * c.n_embd);
    logits_ = dmalloc<float>((size_t)T * c.n_vocab);
    d_sg_ = dmalloc<float>(T);
    cand_tmp_ = dmalloc<float>(128 * kCand * 2);
    d_nmiss_ = dmalloc<int32_t>(1);
    hits_ = dmalloc<HitList>(1);
    xq_ = alloc_act(T * c.n_embd);
    yq_ = alloc_act(T * c.n_head * c.head_dim);   // 4096: delta-net y and attention output
    shq_ = alloc_act(T * c.n_ff_shexp);
    hq_ = alloc_act(kMaxU * c.n_ff_exp);
    route_counts_ = dmalloc<uint32_t>((size_t)NL * c.n_expert);
    tok_counts_ = dmalloc<uint32_t>(c.n_vocab);

    kc_.assign(NL, nullptr);
    vc_.assign(NL, nullptr);
    kc_i8_.assign(NL, nullptr);
    vc_i8_.assign(NL, nullptr);
    kc_scale_.assign(NL, nullptr);
    vc_scale_.assign(NL, nullptr);
    conv_st_.assign(NL, nullptr);
    ssm_st_.assign(NL, nullptr);
    const int64_t conv_f = (int64_t)c.conv_dim() * (c.ssm_d_conv - 1);
    const int64_t ssm_f = (int64_t)c.ssm_n_vh * c.ssm_d_state * c.ssm_d_state;
    state_floats_ = (conv_f + ssm_f) * c.n_gdn_layers();
    state_arena_ = dmalloc<float>(state_floats_);
    ckpts_.resize(std::max(opt_.ckpt_slots, 0));
    for (auto& ck : ckpts_) {
        ck.arena = dmalloc<float>(state_floats_);
        ck.hin = dmalloc<float>((size_t)kMaxT * c.n_embd);
    }
    int64_t off = 0;
    for (int il = 0; il < NL; ++il) {
        if (c.is_attn(il)) {
            if (kv_int8_) {
                kc_i8_[il] = dmalloc<int8_t>((size_t)c.n_head_kv * kv_capacity_ * c.head_dim);
                vc_i8_[il] = dmalloc<int8_t>((size_t)c.n_head_kv * kv_capacity_ * c.head_dim);
                kc_scale_[il] = dmalloc<float>((size_t)c.n_head_kv * kv_capacity_);
                vc_scale_[il] = dmalloc<float>((size_t)c.n_head_kv * kv_capacity_);
            } else {
                kc_[il] = dmalloc<uint16_t>((size_t)c.n_head_kv * kv_capacity_ * c.head_dim);
                vc_[il] = dmalloc<uint16_t>((size_t)c.n_head_kv * kv_capacity_ * c.head_dim);
            }
        } else {
            conv_st_[il] = state_arena_ + off;
            off += conv_f;
            ssm_st_[il] = state_arena_ + off;
            off += ssm_f;
        }
    }
    mtp_hin_ = dmalloc<float>((size_t)kMaxT * c.n_embd);
    const bool draft_subset = env_mib("STRATA_MTP_DRAFT_SUBSET", 0, 1) != 0;
    if (draft_subset) {
        SQ_CHECK(mtp_on_ && !model_.cpu_lm_head && !opt_.use_graph,
                 "STRATA_MTP_DRAFT_SUBSET requires MTP, the GPU LM head, and --no-graph");
        const int per_partition = (int)env_mib("STRATA_MTP_SUBSET_PER_PARTITION", 64, 128);
        mtp_subset_ = std::make_unique<MtpDraftSubset>(c.n_vocab, per_partition);
        log("dynamic MTP draft subset: %d partitions x %d + %d recent slots; full-vocabulary target verification",
            MtpDraftSubset::partitions, per_partition, MtpDraftSubset::recent_slots);
        log("MTP subset confidence is subset-normalized: --draft-p controls a heuristic, not the target sampler");
    }
    if (mtp_on_) {
        mtp_hlast_ = dmalloc<float>(c.n_embd);
        mtp_logits_ = dmalloc<float>(std::max(c.n_vocab, mtp_subset_ ? mtp_subset_->slots() : 0));
        mtp_act_ = alloc_act(T * 2 * c.n_embd);
        snap_arena_ = dmalloc<float>((size_t)opt_.mtp_draft * state_floats_);
    }
    const double kv_gib = kv_bytes() / 1073741824.0;
    log("KV cache %.2f GiB (capacity %d / ctx %d), delta-net state %.0f MiB%s", kv_gib, kv_capacity_, opt_.ctx, state_floats_ * 4 / 1048576.0,
        mtp_on_ ? " (+ rollback snapshots)" : "");
}

void Engine::ensure_kv_capacity(int tokens) {
    SQ_CHECK(tokens >= 0 && tokens <= opt_.ctx, "KV position %d exceeds configured context %d", tokens, opt_.ctx);
    if (tokens <= kv_capacity_) return;
    SQ_CHECK(kv_initial_tokens_ > 0 && !opt_.use_graph && !prefill_,
             "KV growth requires lazy allocation with graphs and large prefill disabled");
    const Config& c = model_.cfg;
    const int previous = kv_capacity_;
    int capacity = previous;
    while (capacity < tokens) capacity = capacity > opt_.ctx / 2 ? opt_.ctx : capacity * 2;
    const int64_t extra = kv_bytes() / previous * (capacity - previous);
    const int64_t cache_before = cache_bytes_;
    const double started = now_ms();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    if (event_graph_moe_) invalidate_event_graphs(); // captured attention nodes contain the old KV addresses/stride
    if (prompt_route_graphs_) invalidate_prompt_route_graphs();
    if (prompt_full_graph_) invalidate_prompt_full_graphs();
    // Initial lazy allocation gave these bytes to experts.  Return at least this much before growing KV,
    // so a populated expert cache cannot cause a full-context-size allocation on top of the old VRAM budget.
    if (cache_bytes_ > 0) trim_cache(std::min(extra, cache_bytes_));
    const size_t max_token_bytes = std::max<size_t>(c.head_dim * (kv_int8_ ? sizeof(int8_t) : sizeof(uint16_t)),
                                                   kv_int8_ ? sizeof(float) : 0);
    const size_t staging_bytes = (size_t)c.n_head_kv * previous * max_token_bytes;
    uint8_t* staging = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&staging, staging_bytes, cudaHostAllocDefault));
    auto grow_buffer = [&]<typename Element>(Element*& buffer, int elements_per_token) {
        if (!buffer) return;
        const size_t old_pitch = (size_t)previous * elements_per_token * sizeof(Element);
        const size_t new_pitch = (size_t)capacity * elements_per_token * sizeof(Element);
        const size_t old_bytes = old_pitch * c.n_head_kv, new_bytes = new_pitch * c.n_head_kv;
        SQ_CHECK(old_bytes <= staging_bytes, "KV staging buffer is smaller than a head-row copy");
        // Preserve every old position, including pending MTP writes/checkpoint suffixes.  Only one K, V, or
        // scale buffer is staged at a time, and its old device allocation is freed before the larger one exists.
        CUDA_CHECK(cudaMemcpyAsync(staging, buffer, old_bytes, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFree(buffer));
        buffer = nullptr;
        cudaError_t allocation = cudaMalloc(&buffer, new_bytes);
        if (allocation != cudaSuccess && cache_bytes_ > 0) {
            cudaGetLastError();
            buffer = nullptr;
            // Another application or fragmentation may have consumed the planned headroom.  Give up more
            // expert slots once; never silently lower the configured context or discard the preserved KV rows.
            trim_cache(std::min<int64_t>((int64_t)new_bytes, cache_bytes_));
            allocation = cudaMalloc(&buffer, new_bytes);
        }
        SQ_CHECK(allocation == cudaSuccess, "cannot grow KV capacity %d -> %d (context %d): %s",
                 previous, capacity, opt_.ctx, cudaGetErrorString(allocation));
        CUDA_CHECK(cudaMemsetAsync(buffer, 0, new_bytes, stream_));
        CUDA_CHECK(cudaMemcpy2DAsync(buffer, new_pitch, staging, old_pitch, old_pitch, c.n_head_kv,
                                     cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_)); // pinned staging is reused by the next buffer
    };
    for (int layer = 0; layer < c.n_moe_layers(); ++layer) {
        if (kv_int8_) {
            grow_buffer(kc_i8_[layer], c.head_dim);
            grow_buffer(vc_i8_[layer], c.head_dim);
            grow_buffer(kc_scale_[layer], 1);
            grow_buffer(vc_scale_[layer], 1);
        } else {
            grow_buffer(kc_[layer], c.head_dim);
            grow_buffer(vc_[layer], c.head_dim);
        }
    }
    CUDA_CHECK(cudaFreeHost(staging));
    kv_capacity_ = capacity;
    log("KV capacity grew %d -> %d / %d tokens: +%.1f MiB, expert cache %.1f -> %.1f MiB, %.1f ms",
        previous, capacity, opt_.ctx, extra / 1048576.0, cache_before / 1048576.0, cache_bytes_ / 1048576.0,
        now_ms() - started);
}

// ================================================================================================= expert cache
void Engine::setup_cache() {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers(), NE = c.n_expert;
    freq_.assign((size_t)NL * NE, 0.0);
    counts_seen_.assign((size_t)NL * NE, 0);
    route_host_.assign((size_t)NL * NE, 0);
    prior_.assign((size_t)NL * NE, 0.0f);
    bool have_profile = false;
    if (!opt_.profile_path.empty()) {
        std::ifstream f(opt_.profile_path, std::ios::binary);
        uint32_t hdr[3] = {0, 0, 0};
        // profiles written without / with the MTP layer both fit: the layers they share are used
        if (f && f.read((char*)hdr, sizeof(hdr)) && hdr[0] == kProfileMagic && (int)hdr[1] >= c.n_layer &&
            (int)hdr[1] <= c.n_layer + 1 && (int)hdr[2] == NE) {
            const int FL = (int)hdr[1];
            std::vector<float> fr((size_t)FL * NE);
            if (f.read((char*)fr.data(), fr.size() * 4)) {
                // The profile is a long-run count; scale each layer to profile_weight times the mass the decayed live
                // counts settle at (n_expert_used * adapt_every / (1 - decay)), so that it is a prior the current
                // conversation can override within a few hundred tokens instead of freezing the cache.
                const double live = (double)c.n_expert_used * std::max(opt_.adapt_every, 1) / (1.0 - opt_.adapt_decay);
                for (int l = 0; l < std::min(NL, FL); ++l) {
                    double sum = 0;
                    for (int e = 0; e < NE; ++e) sum += fr[(size_t)l * NE + e];
                    const double k = sum > 0 ? opt_.profile_weight * live / sum : 0.0;
                    for (int e = 0; e < NE; ++e) {
                        const size_t i = (size_t)l * NE + e;
                        prior_[i] = fr[i];
                        freq_[i] = fr[i] * k;
                    }
                }
                have_profile = true;
                log("expert profile: %s", opt_.profile_path.c_str());
            }
        }
        if (!have_profile) log("expert profile %s not usable; starting from a uniform cache", opt_.profile_path.c_str());
    }
    size_t fr = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    // What the warm-up and graph capture still allocate after this point (init() trims the cache if
    // it turns out to be more).  Without it that memory would come out of the reserve, and under WDDM an
    // over-full card spills to system memory instead of failing.
    const int64_t init_reserve_mib = env_mib("STRATA_CACHE_INIT_RESERVE_MIB", 768, 4096);
    int64_t budget = (int64_t)fr + cache_budget_extra_mib_ * 1048576ll
                     - opt_.vram_reserve_mb * 1048576ll - init_reserve_mib * 1048576ll;
    if (opt_.cache_mb >= 0) budget = std::min<int64_t>(budget, opt_.cache_mb * 1048576ll);
    budget = std::max<int64_t>(budget, 0);
    log("expert cache budget: actual CUDA free %.1f MiB, extra allowance %lld MiB, "
        "startup reserve %lld MiB, runtime reserve %lld MiB, "
        "requested cap %lld MiB (negative=auto), budget %.1f MiB",
        fr / 1048576.0, (long long)cache_budget_extra_mib_, (long long)init_reserve_mib, (long long)opt_.vram_reserve_mb,
        (long long)opt_.cache_mb, budget / 1048576.0);

    // Rank (layer, expert) by frequency; take greedily while the bytes fit.  Without a profile every layer gets
    // the same share (frequencies are all zero; the tie-break spreads the slots across layers).
    std::vector<int> order((size_t)NL * NE);
    std::iota(order.begin(), order.end(), 0);
    std::vector<int> rank_in_layer((size_t)NL * NE);
    for (int i = 0; i < NL * NE; ++i) rank_in_layer[i] = i % NE;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (freq_[a] != freq_[b]) return freq_[a] > freq_[b];
        return rank_in_layer[a] < rank_in_layer[b];
    });
    slots_per_layer_.assign(NL, 0);
    int64_t used = 0;
    std::vector<std::vector<int>> chosen(NL);
    for (int idx : order) {
        const int l = idx / NE;
        const int64_t b = model_.layers[l].blob_bytes;
        if (used + b > budget) continue;
        used += b;
        chosen[l].push_back(idx % NE);
    }
    residency_host_.assign((size_t)NL * NE, -1);
    slot_base_.assign(NL, nullptr);
    slot_expert_.assign(NL, {});
    std::vector<CacheLayerInfo> ci(NL);
    cache_slots_total_ = 0;
    cache_bytes_ = 0;
    for (int l = 0; l < NL; ++l) {
        const int n = (int)chosen[l].size();
        slots_per_layer_[l] = n;
        const int64_t b = model_.layers[l].blob_bytes;
        if (n > 0) {
            cudaError_t e = cudaMalloc(&slot_base_[l], (size_t)(b * n));
            if (e != cudaSuccess) {   // fragmentation or another program took memory: shrink this layer
                cudaGetLastError();
                slots_per_layer_[l] = 0;
                chosen[l].clear();
                continue;
            }
        }
        ci[l] = CacheLayerInfo{slot_base_[l], (long long)b};
        for (int s = 0; s < n; ++s) {
            const int e = chosen[l][s];
            residency_host_[(size_t)l * NE + e] = s;
            slot_expert_[l].push_back(e);
            CUDA_CHECK(cudaMemcpyAsync(slot_base_[l] + s * b, model_.layers[l].expert_blob(e), (size_t)b,
                                       cudaMemcpyHostToDevice, stream_));
        }
        cache_slots_total_ += n;
        cache_bytes_ += b * n;
    }
    residency_dev_ = dmalloc<int32_t>((size_t)NL * NE);
    cinfo_dev_ = dmalloc<CacheLayerInfo>(NL);
    CUDA_CHECK(cudaMemcpyAsync(residency_dev_, residency_host_.data(), residency_host_.size() * 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(cinfo_dev_, ci.data(), ci.size() * sizeof(CacheLayerInfo), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Engine::setup_miss_stream() {
    const int NL = model_.cfg.n_moe_layers(), NE = model_.cfg.n_expert;
    miss_stream_eligible_.assign(NL, {});
    int selected = 0, positive_selected = 0, zero_profile_layers = 0;
    for (int layer = 0; layer < NL; ++layer) {
        if (!model_.layers[layer].experts_pinned) continue; // no implicit pageable/staging copy in this opt-in
        std::vector<int> order(NE);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return prior_[(size_t)layer * NE + a] > prior_[(size_t)layer * NE + b];
        });
        auto& eligible = miss_stream_eligible_[layer];
        int positive_in_layer = 0;
        for (int expert : order) positive_in_layer += prior_[(size_t)layer * NE + expert] > 0;
        zero_profile_layers += positive_in_layer == 0;
        for (int expert : order) {
            if (residency_host_[(size_t)layer * NE + expert] >= 0) continue;
            eligible.push_back(expert);
            positive_selected += prior_[(size_t)layer * NE + expert] > 0;
            if ((int)eligible.size() == miss_stream_eligible_limit_) break;
        }
        std::sort(eligible.begin(), eligible.end());
        selected += (int)eligible.size();
        if (layer >= model_.cfg.n_layer) {
            std::string ids;
            for (int expert : eligible) ids += (ids.empty() ? "" : ",") + std::to_string(expert);
            log("transient MTP layer %d: %d experts with positive loaded profile counts; fixed eligible IDs [%s]",
                layer, positive_in_layer, ids.c_str());
        }
    }
    log("transient eligibility frozen from initial nonresident profile ranking: %d IDs across %d layers; no per-window reselection",
        selected, NL);
    log("transient profile evidence: %d/%d selected IDs have positive counts, %d pinned layers have no positive counts; ties use ascending expert ID",
        positive_selected, selected, zero_profile_layers);
}

int Engine::prepare_miss_stream(int layer, int T, cudaStream_t s) {
    Mailbox& mailbox = mb_host_[layer];
    const auto& eligible = miss_stream_eligible_[layer];
    if (eligible.empty() || mailbox.n_miss == 0) return 0;
    const LayerW& weights = model_.layers[layer];
    SQ_CHECK(weights.experts_pinned && (size_t)weights.blob_bytes <= miss_stream_stride_,
             "invalid transient expert source or staging stride");
    HitList& plan = *miss_stream_hits_host_;
    std::memset(&plan, 0, sizeof(plan));
    std::memset(plan.pair, -1, sizeof(plan.pair));
    int kept = 0;
    for (int j = 0; j < mailbox.n_miss; ++j) {
        const int id = mailbox.ids[j];
        if (!std::binary_search(eligible.begin(), eligible.end(), id)) {
            mailbox.ids[kept] = id;
            if (kept != j) std::memcpy(mailbox.w[kept], mailbox.w[j], sizeof(mailbox.w[j]));
            ++kept;
            continue;
        }
        SQ_CHECK(plan.n < miss_stream_capacity_ && plan.n < kMaxU && plan.n < kMaxK * T,
                 "transient expert capacity exceeded; eligible routes must never be truncated");
        if (plan.n == 0) miss_stream_last_ids_.clear();
        const int slot = plan.n++;
        uint8_t* destination = miss_stream_bank_ + (size_t)slot * miss_stream_stride_;
        plan.ptr[slot] = destination;
        for (int token = 0; token < T; ++token) {
            plan.w[slot][token] = mailbox.w[j][token];
            if (T == 1 || mailbox.w[j][token] != 0.0f) {
                SQ_CHECK(plan.n_pair < kMaxU && plan.n_pair < kMaxK * T, "transient expert/token capacity exceeded");
                plan.pair[slot][token] = (int8_t)plan.n_pair++;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(destination, weights.expert_blob(id), (size_t)weights.blob_bytes,
                                   cudaMemcpyHostToDevice, miss_copy_stream_));
        miss_stream_last_ids_.push_back(id);
    }
    if (plan.n == 0) return 0;
    mailbox.n_miss = kept;
    miss_stream_last_layer_ = layer;
    // This metadata and all source blobs remain immutable until the next router-ready event, which is after
    // this layer's streamed expert kernels and combine. No persistent residency or route statistics are changed.
    CUDA_CHECK(cudaMemcpyAsync(miss_stream_hits_dev_, &plan, sizeof(plan), cudaMemcpyHostToDevice, miss_copy_stream_));
    CUDA_CHECK(cudaEventRecord(miss_stream_ready_, miss_copy_stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_nmiss_, &mailbox.n_miss, sizeof(mailbox.n_miss), cudaMemcpyHostToDevice, s));
    if (miss_stream_serialized_) CUDA_CHECK(cudaEventSynchronize(miss_stream_ready_));
    stats.streamed_experts += plan.n;
    stats.streamed_pairs += plan.n_pair;
    stats.streamed_bytes += (uint64_t)plan.n * weights.blob_bytes;
    stats.stream_max_unique = std::max(stats.stream_max_unique, (uint64_t)plan.n);
    stats.stream_max_pairs = std::max(stats.stream_max_pairs, (uint64_t)plan.n_pair);
    stats.stream_zero_cpu_layers += kept == 0;
    return plan.n;
}

void Engine::setup_async_cache() {
    const int NL = model_.cfg.n_moe_layers(), NE = model_.cfg.n_expert;
    int64_t largest_blob = 0;
    for (int l = 0; l < NL; ++l)
        largest_blob = std::max(largest_blob, model_.layers[l].blob_bytes);
    // adapt_cache considers at most eight replacements in each layer, even if --adapt-swaps is larger.
    const int staging_slots = (int)std::min<int64_t>(cache_slots_total_,
        std::max(rebalance_every_ > 0 ? 1 : 0, std::clamp(opt_.adapt_swaps, 0, NL * 8)));
    cache_upload_capacity_ = (size_t)staging_slots * (size_t)largest_blob;
    if (cache_upload_capacity_ > 0)
        CUDA_CHECK(cudaHostAlloc((void**)&cache_upload_host_, cache_upload_capacity_, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&cache_residency_host_, (size_t)NL * NE * sizeof(int32_t), cudaHostAllocDefault));
    cache_layer_ready_.assign(NL, nullptr);
    for (auto& event : cache_layer_ready_) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(event, copy_stream_));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&cache_upload_done_, cudaEventBlockingSync |
                                       (rebalance_every_ > 0 ? 0 : cudaEventDisableTiming)));
    if (rebalance_every_ > 0)
        CUDA_CHECK(cudaEventCreate(&cache_upload_started_));
    CUDA_CHECK(cudaEventRecord(cache_upload_done_, copy_stream_));
    log("expert cache upload staging: %.1f MiB pinned, up to %d swaps per round",
        cache_upload_capacity_ / 1048576.0, staging_slots);
}

void Engine::apply_swap(int layer, int slot, int expert) {
    const int NE = model_.cfg.n_expert;
    const int64_t b = model_.layers[layer].blob_bytes;
    const int old = slot_expert_[layer][slot];
    residency_host_[(size_t)layer * NE + old] = -1;
    residency_host_[(size_t)layer * NE + expert] = slot;
    slot_expert_[layer][slot] = expert;
    CUDA_CHECK(cudaMemcpyAsync(slot_base_[layer] + slot * b, model_.layers[layer].expert_blob(expert), (size_t)b,
                               cudaMemcpyHostToDevice, stream_));
    ++stats.swaps;
}

// Shrinks the cache in place: the coldest resident experts go, and each layer that loses slots gets a smaller slab
// (freed first, so this works on a full card) with its remaining experts uploaded again from host memory.  Only
// between requests: nothing may be in flight on the GPU or the CPU workers.
void Engine::trim_cache(int64_t bytes) {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers(), NE = c.n_expert;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    std::vector<int> res;   // resident (layer, expert), coldest first
    for (int l = 0; l < NL; ++l)
        for (int e : slot_expert_[l]) res.push_back(l * NE + e);
    std::stable_sort(res.begin(), res.end(), [&](int a, int b) { return freq_[a] < freq_[b]; });
    std::vector<char> drop((size_t)NL * NE, 0), touched(NL, 0);
    int64_t freed = 0;
    int dropped = 0;
    for (int idx : res) {
        if (freed >= bytes) break;
        drop[idx] = 1;
        touched[idx / NE] = 1;
        freed += model_.layers[idx / NE].blob_bytes;
        ++dropped;
    }
    for (int l = 0; l < NL; ++l) {
        if (!touched[l]) continue;
        const int64_t b = model_.layers[l].blob_bytes;
        std::vector<int> keep;
        for (int e : slot_expert_[l]) {
            residency_host_[(size_t)l * NE + e] = -1;
            if (!drop[(size_t)l * NE + e]) keep.push_back(e);
        }
        CUDA_CHECK(cudaFree(slot_base_[l]));
        slot_base_[l] = nullptr;
        cache_bytes_ -= b * slots_per_layer_[l];
        cache_slots_total_ -= slots_per_layer_[l];
        if (!keep.empty() && cudaMalloc(&slot_base_[l], (size_t)(b * keep.size())) != cudaSuccess) {
            cudaGetLastError();
            slot_base_[l] = nullptr;
            keep.clear();
        }
        slot_expert_[l] = keep;
        slots_per_layer_[l] = (int)keep.size();
        for (int s = 0; s < (int)keep.size(); ++s) {
            residency_host_[(size_t)l * NE + keep[s]] = s;
            CUDA_CHECK(cudaMemcpyAsync(slot_base_[l] + s * b, model_.layers[l].expert_blob(keep[s]), (size_t)b,
                                       cudaMemcpyHostToDevice, stream_));
        }
        cache_bytes_ += b * slots_per_layer_[l];
        cache_slots_total_ += slots_per_layer_[l];
    }
    std::vector<CacheLayerInfo> ci(NL);
    for (int l = 0; l < NL; ++l) ci[l] = CacheLayerInfo{slot_base_[l], (long long)model_.layers[l].blob_bytes};
    CUDA_CHECK(cudaMemcpyAsync(residency_dev_, residency_host_.data(), residency_host_.size() * 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(cinfo_dev_, ci.data(), ci.size() * sizeof(CacheLayerInfo), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    log("expert cache trimmed by %d slots (%.0f MiB) toward %lld MiB runtime reserve (extra allowance %lld MiB)",
        dropped, freed / 1048576.0, (long long)opt_.vram_reserve_mb, (long long)cache_budget_extra_mib_);
}

// Rebuild the cache after the prompt path borrowed its VRAM. Capacities are the post-KV-growth per-layer slot
// counts. Prompt routes are folded into the normal decayed frequency before choosing each layer's hottest experts.
void Engine::refill_cache_after_prefill(const std::vector<std::vector<int>>& previous_layout) {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers(), NE = c.n_expert;
    SQ_CHECK((int)previous_layout.size() == NL && cache_bytes_ == 0, "invalid phase-switched cache refill state");
    CUDA_CHECK(cudaMemcpyAsync(route_host_.data(), route_counts_, route_host_.size() * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    // The bounded decode path adapts every `adapt_every` tokens, so one update contributes only K*adapt_every
    // routes per layer. A large prefill supplies hundreds or thousands of tokens at once; adding that raw total
    // would erase the long-run profile and overfit the cache to the whole prompt. Preserve its distribution while
    // normalizing its mass to one ordinary adaptation round.
    std::vector<uint64_t> layer_delta(NL, 0);
    for (int layer = 0; layer < NL; ++layer)
        for (int expert = 0; expert < NE; ++expert)
            layer_delta[layer] += route_host_[(size_t)layer * NE + expert] -
                                  counts_seen_[(size_t)layer * NE + expert];
    const double round_mass = (double)c.n_expert_used * std::max(opt_.adapt_every, 1);
    for (size_t i = 0; i < freq_.size(); ++i) {
        const uint32_t delta = route_host_[i] - counts_seen_[i];
        counts_seen_[i] = route_host_[i];
        const uint64_t total = layer_delta[i / NE];
        const double normalized = total > 0 ? delta * round_mass / total : 0.0;
        freq_[i] = freq_[i] * opt_.adapt_decay + normalized;
    }
    std::fill(residency_host_.begin(), residency_host_.end(), -1);
    cache_slots_total_ = 0;
    std::vector<CacheLayerInfo> info(NL);
    uint64_t uploaded = 0;
    for (int layer = 0; layer < NL; ++layer) {
        const int64_t blob = model_.layers[layer].blob_bytes;
        std::vector<int> order = previous_layout[layer];
        int count = std::min((int)order.size(), NE);
        while (count > 0 && cudaMalloc(&slot_base_[layer], (size_t)count * blob) != cudaSuccess) {
            cudaGetLastError();
            slot_base_[layer] = nullptr;
            --count;
        }
        slots_per_layer_[layer] = count;
        slot_expert_[layer].assign(order.begin(), order.begin() + count);
        for (int slot = 0; slot < count; ++slot) {
            const int expert = slot_expert_[layer][slot];
            residency_host_[(size_t)layer * NE + expert] = slot;
            CUDA_CHECK(cudaMemcpyAsync(slot_base_[layer] + (size_t)slot * blob,
                                       model_.layers[layer].expert_blob(expert), (size_t)blob,
                                       cudaMemcpyHostToDevice, stream_));
            uploaded += (uint64_t)blob;
        }
        cache_slots_total_ += count;
        cache_bytes_ += (int64_t)count * blob;
        info[layer] = CacheLayerInfo{slot_base_[layer], (long long)blob};
    }
    CUDA_CHECK(cudaMemcpyAsync(residency_dev_, residency_host_.data(), residency_host_.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(cinfo_dev_, info.data(), info.size() * sizeof(CacheLayerInfo),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (cache_residency_host_)
        std::memcpy(cache_residency_host_, residency_host_.data(), residency_host_.size() * sizeof(int32_t));
    log("phase-switched prefill: refilled %lld expert slots (%.1f MiB) from prompt routing, uploaded %.1f MiB",
        (long long)cache_slots_total_, cache_bytes_ / 1048576.0, uploaded / 1048576.0);
}

void Engine::rebalance_cache() {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers(), NE = c.n_expert;
    if (cache_bytes_ <= 0 || cache_upload_capacity_ == 0 || cache_upload_bytes_per_ms_ <= 0) return;
    // The caller has drained compute and the preceding upload round.  Rank the whole cache by score/byte,
    // but implement only one bounded donor/receiver move, not a disruptive whole-cache reconstruction.
    std::vector<int> ranked((size_t)NL * NE), target(NL, 0);
    std::iota(ranked.begin(), ranked.end(), 0);
    std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) {
        return freq_[a] / model_.layers[a / NE].blob_bytes > freq_[b] / model_.layers[b / NE].blob_bytes;
    });
    int64_t available = cache_bytes_;
    for (int index : ranked) {
        const int layer = index / NE;
        const int64_t bytes = model_.layers[layer].blob_bytes;
        if (bytes <= available) { ++target[layer]; available -= bytes; }
    }
    std::vector<std::vector<int>> cold(NL), hot(NL);
    std::vector<double> mass(NL, 0);
    for (int layer = 0; layer < NL; ++layer) {
        cold[layer] = slot_expert_[layer];
        const double* f = freq_.data() + (size_t)layer * NE;
        std::sort(cold[layer].begin(), cold[layer].end(), [&](int a, int b) { return f[a] < f[b]; });
        for (int expert = 0; expert < NE; ++expert) {
            mass[layer] += f[expert];
            if (residency_host_[(size_t)layer * NE + expert] < 0) hot[layer].push_back(expert);
        }
        std::sort(hot[layer].begin(), hot[layer].end(), [&](int a, int b) { return f[a] > f[b]; });
    }
    struct Move { int donor = -1, receiver = -1, drop = 0, add = 0; double benefit = 0, cost = 0, margin = 0; } best;
    for (int receiver = 0; receiver < NL; ++receiver) {
        const int want = std::min(8, target[receiver] - slots_per_layer_[receiver]);
        if (want <= 0 || mass[receiver] <= 0 || cpu_pair_ms_[receiver] <= 0) continue;
        const int64_t rb = model_.layers[receiver].blob_bytes;
        double add_rate = 0;
        for (int add = 1; add <= want && add <= (int)hot[receiver].size(); ++add) {
            add_rate += c.n_expert_used * freq_[(size_t)receiver * NE + hot[receiver][add - 1]] / mass[receiver];
            for (int donor = 0; donor < NL; ++donor) {
                if (donor == receiver || mass[donor] <= 0 || cpu_pair_ms_[donor] <= 0) continue;
                const int64_t db = model_.layers[donor].blob_bytes;
                const int drop = (int)(((int64_t)add * rb + db - 1) / db);
                if (drop > 8 || drop > slots_per_layer_[donor] - target[donor]) continue;
                const int64_t copied = (slots_per_layer_[donor] - drop) * db +
                                       (slots_per_layer_[receiver] + add) * rb;
                const int64_t restore_bytes = slots_per_layer_[donor] * db + slots_per_layer_[receiver] * rb;
                if (std::max(copied, restore_bytes) > rebalance_max_bytes_) continue;
                double drop_rate = 0;
                for (int j = 0; j < drop; ++j)
                    drop_rate += c.n_expert_used * freq_[(size_t)donor * NE + cold[donor][j]] / mass[donor];
                if (add_rate <= drop_rate) continue;
                // CPU job timing excludes the LM head.  Halve the estimated saved work for GPU execution and
                // overlap, then demand another 2x margin over observed copy cost plus allocation overhead.
                const double benefit = 0.5 * rebalance_every_ *
                    (add_rate * cpu_pair_ms_[receiver] - drop_rate * cpu_pair_ms_[donor]);
                const double cost = std::max(4.0 + copied / cache_upload_bytes_per_ms_, copied * rebalance_cost_per_byte_);
                if (std::isfinite(benefit) && benefit - 2.0 * cost > best.margin)
                    best = {donor, receiver, drop, add, benefit, cost, benefit - 2.0 * cost};
            }
        }
    }
    if (best.donor < 0) return;
    const double started = now_ms();
    const int layers[2] = {best.donor, best.receiver};
    const std::vector<int> old[2] = {slot_expert_[layers[0]], slot_expert_[layers[1]]};
    std::vector<int> desired[2];
    desired[0].assign(cold[best.donor].begin() + best.drop, cold[best.donor].end());
    desired[1] = old[1];
    desired[1].insert(desired[1].end(), hot[best.receiver].begin(), hot[best.receiver].begin() + best.add);
    const int64_t old_budget = cache_bytes_;
    for (int k = 0; k < 2; ++k) {
        const int layer = layers[k];
        CUDA_CHECK(cudaFree(slot_base_[layer]));
        slot_base_[layer] = nullptr;
        cache_bytes_ -= (int64_t)old[k].size() * model_.layers[layer].blob_bytes;
        cache_slots_total_ -= old[k].size();
    }
    bool restored = false;
    for (int k = 0; k < 2; ++k) {
        const int layer = layers[k];
        if (!desired[k].empty() && cudaMalloc(&slot_base_[layer],
            desired[k].size() * (size_t)model_.layers[layer].blob_bytes) != cudaSuccess) {
            cudaGetLastError();
            restored = true;
            break;
        }
    }
    if (restored) {
        // Release any partial new layout before restoring.  External allocation pressure can even prevent
        // restoration; halve each slab until it fits, keeping the hottest old experts and valid metadata.
        for (int layer : layers) { if (slot_base_[layer]) CUDA_CHECK(cudaFree(slot_base_[layer])); slot_base_[layer] = nullptr; }
        for (int k = 0; k < 2; ++k) {
            const int layer = layers[k];
            desired[k] = old[k];
            std::sort(desired[k].begin(), desired[k].end(), [&](int a, int b) {
                return freq_[(size_t)layer * NE + a] > freq_[(size_t)layer * NE + b];
            });
            while (!desired[k].empty() && cudaMalloc(&slot_base_[layer],
                desired[k].size() * (size_t)model_.layers[layer].blob_bytes) != cudaSuccess) {
                cudaGetLastError();
                slot_base_[layer] = nullptr;
                desired[k].resize(desired[k].size() / 2);
            }
        }
    }
    size_t offset = 0, uploaded = 0;
    CacheLayerInfo info[2];
    for (int k = 0; k < 2; ++k) {
        const int layer = layers[k];
        const int64_t blob = model_.layers[layer].blob_bytes;
        std::fill_n(residency_host_.data() + (size_t)layer * NE, NE, -1);
        slot_expert_[layer] = desired[k];
        slots_per_layer_[layer] = (int)desired[k].size();
        cache_bytes_ += (int64_t)desired[k].size() * blob;
        cache_slots_total_ += desired[k].size();
        for (int slot = 0; slot < (int)desired[k].size(); ++slot) {
            if (offset + (size_t)blob > cache_upload_capacity_) {
                CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
                offset = 0;
            }
            const int expert = desired[k][slot];
            std::memcpy(cache_upload_host_ + offset, model_.layers[layer].expert_blob(expert), (size_t)blob);
            CUDA_CHECK(cudaMemcpyAsync(slot_base_[layer] + slot * blob, cache_upload_host_ + offset, (size_t)blob,
                                       cudaMemcpyHostToDevice, copy_stream_));
            offset += (size_t)blob;
            uploaded += (size_t)blob;
            residency_host_[(size_t)layer * NE + expert] = slot;
        }
        int32_t* snapshot = cache_residency_host_ + (size_t)layer * NE;
        std::memcpy(snapshot, residency_host_.data() + (size_t)layer * NE, NE * sizeof(int32_t));
        CUDA_CHECK(cudaMemcpyAsync(residency_dev_ + (size_t)layer * NE, snapshot, NE * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, copy_stream_));
        info[k] = CacheLayerInfo{slot_base_[layer], (long long)blob};
        CUDA_CHECK(cudaMemcpyAsync(cinfo_dev_ + layer, info + k, sizeof(CacheLayerInfo), cudaMemcpyHostToDevice, copy_stream_));
        CUDA_CHECK(cudaEventRecord(cache_layer_ready_[layer], copy_stream_));
    }
    CUDA_CHECK(cudaEventRecord(cache_upload_done_, copy_stream_));
    CUDA_CHECK(cudaEventSynchronize(cache_upload_done_));
    SQ_CHECK(cache_bytes_ <= old_budget, "cache rebalance exceeded its original byte budget");
    const double elapsed = now_ms() - started;
    if (uploaded > 0) rebalance_cost_per_byte_ = elapsed / uploaded;
    ++stats.cache_rebalances;
    stats.rebalance_slots += restored ? 0 : best.add;
    stats.rebalance_bytes += uploaded;
    stats.rebalance_ms += elapsed;
    stats.swaps += restored ? 0 : best.add;
    log("expert cache rebalance L%d->L%d: -%d/+%d slots%s, %.1f MiB uploaded in %.2f ms; "
        "estimated benefit %.2f ms / cost %.2f ms over %d tokens; total %lld slots %.1f MiB",
        best.donor, best.receiver, best.drop, best.add, restored ? " (allocation fallback)" : "",
        uploaded / 1048576.0, elapsed, best.benefit, best.cost, rebalance_every_,
        (long long)cache_slots_total_, cache_bytes_ / 1048576.0);
}

void Engine::adapt_cache() {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers(), NE = c.n_expert;
    // An earlier round may still be copying a layer that the last step did not use (e.g. the MTP layer).
    // Do not modify pinned staging, residency snapshots, or an in-flight cache slot until that round finishes.
    if (async_cache_) {
        CUDA_CHECK(cudaEventSynchronize(cache_upload_done_));
        if (rebalance_every_ > 0 && cache_upload_bytes_ > 0) {
            float elapsed = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, cache_upload_started_, cache_upload_done_));
            if (elapsed > 0 && std::isfinite(elapsed)) {
                const double rate = cache_upload_bytes_ / (double)elapsed;
                cache_upload_bytes_per_ms_ = cache_upload_bytes_per_ms_ > 0 ?
                    0.75 * cache_upload_bytes_per_ms_ + 0.25 * rate : rate;
            }
            cache_upload_bytes_ = 0;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(route_host_.data(), route_counts_, route_host_.size() * 4, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    // Exponentially decayed routing frequency: the profile fades, the conversation takes over.
    uint64_t trunk_routes = 0;
    for (size_t i = 0; i < freq_.size(); ++i) {
        const uint32_t d = route_host_[i] - counts_seen_[i];
        if (i < (size_t)NE) trunk_routes += d;
        counts_seen_[i] = route_host_[i];
        freq_[i] = freq_[i] * opt_.adapt_decay + d;
    }
    if (rebalance_every_ > 0) {
        rebalance_tokens_ += trunk_routes / c.n_expert_used;
        if (rebalance_tokens_ >= (uint64_t)rebalance_every_) {
            rebalance_tokens_ %= rebalance_every_;
            rebalance_cache();
        }
    }
    // Candidate swaps: across layers, the hottest non-resident expert vs its layer's coldest resident one.
    struct Cand { double gain; int layer, slot, expert; };
    std::vector<Cand> cands;
    for (int l = 0; l < NL; ++l) {
        const int n = slots_per_layer_[l];
        if (n == 0) continue;
        std::vector<int> res(slot_expert_[l]);
        std::vector<int> non;
        for (int e = 0; e < NE; ++e)
            if (residency_host_[(size_t)l * NE + e] < 0) non.push_back(e);
        const double* F = freq_.data() + (size_t)l * NE;
        std::sort(non.begin(), non.end(), [&](int a, int b) { return F[a] > F[b]; });
        std::vector<int> slots(n);
        std::iota(slots.begin(), slots.end(), 0);
        std::sort(slots.begin(), slots.end(), [&](int a, int b) { return F[slot_expert_[l][a]] < F[slot_expert_[l][b]]; });
        for (size_t k = 0; k < non.size() && k < slots.size() && k < 8; ++k) {
            const double gain = F[non[k]] - F[slot_expert_[l][slots[k]]];
            if (gain <= 1.0) break;   // hysteresis: only clear wins
            cands.push_back({gain, l, slots[k], non[k]});
        }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.gain > b.gain; });
    if (async_cache_) {
        if (rebalance_every_ > 0) CUDA_CHECK(cudaEventRecord(cache_upload_started_, copy_stream_));
        std::vector<Cand> selected;
        for (const Cand& cd : cands) {
            if ((int)selected.size() >= opt_.adapt_swaps) break;
            if (residency_host_[(size_t)cd.layer * NE + cd.expert] >= 0) continue;
            selected.push_back(cd);
        }
        // Upload early trunk layers first.  The next step can use them while later layers are still copying.
        std::stable_sort(selected.begin(), selected.end(), [](const Cand& a, const Cand& b) { return a.layer < b.layer; });
        size_t offset = 0;
        for (size_t i = 0; i < selected.size();) {
            const int layer = selected[i].layer;
            const int64_t blob = model_.layers[layer].blob_bytes;
            do {
                const Cand& cd = selected[i];
                SQ_CHECK(offset + (size_t)blob <= cache_upload_capacity_, "expert upload exceeds pinned staging capacity");
                const int old = slot_expert_[layer][cd.slot];
                residency_host_[(size_t)layer * NE + old] = -1;
                residency_host_[(size_t)layer * NE + cd.expert] = cd.slot;
                slot_expert_[layer][cd.slot] = cd.expert;
                const LayerW& weights = model_.layers[layer];
                const uint8_t* source = weights.expert_blob(cd.expert);
                const bool direct = direct_pinned_cache_ && weights.experts_pinned;
                if (!direct) {
                    // Pageable/file-mapped models still need bounded pinned staging. Keep the same capacity
                    // accounting even for direct uploads, so mixed storage cannot overrun the fallback.
                    std::memcpy(cache_upload_host_ + offset, source, (size_t)blob);
                    source = cache_upload_host_ + offset;
                }
                // Direct sources are immutable after model load. Engine destruction drains copy_stream_
                // before Model releases the pinned arenas, so every source outlives its queued DMA.
                CUDA_CHECK(cudaMemcpyAsync(slot_base_[layer] + cd.slot * blob, source,
                                           (size_t)blob, cudaMemcpyHostToDevice, copy_stream_));
                if (direct) ++direct_pinned_uploads_;
                offset += (size_t)blob;
                ++stats.swaps;
                ++i;
            } while (i < selected.size() && selected[i].layer == layer);
            int32_t* snapshot = cache_residency_host_ + (size_t)layer * NE;
            std::memcpy(snapshot, residency_host_.data() + (size_t)layer * NE, (size_t)NE * sizeof(int32_t));
            // Publish the new residency only after every replacement blob in this layer has arrived.
            CUDA_CHECK(cudaMemcpyAsync(residency_dev_ + (size_t)layer * NE, snapshot, (size_t)NE * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, copy_stream_));
            CUDA_CHECK(cudaEventRecord(cache_layer_ready_[layer], copy_stream_));
        }
        CUDA_CHECK(cudaEventRecord(cache_upload_done_, copy_stream_));
        cache_upload_bytes_ = offset;
        return;  // waits occur at the consuming layer, not globally at the end of adaptation
    }
    int done = 0;
    for (const Cand& cd : cands) {
        if (done >= opt_.adapt_swaps) break;
        if (residency_host_[(size_t)cd.layer * NE + cd.expert] >= 0) continue;
        apply_swap(cd.layer, cd.slot, cd.expert);
        ++done;
    }
    if (done > 0)
        CUDA_CHECK(cudaMemcpyAsync(residency_dev_, residency_host_.data(), residency_host_.size() * 4, cudaMemcpyHostToDevice, stream_));
    // the residency upload reads pageable host memory: make it complete before the table changes again
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

bool Engine::save_profile(const std::string& path, std::string& err) {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers();
    CUDA_CHECK(cudaMemcpy(route_host_.data(), route_counts_, route_host_.size() * 4, cudaMemcpyDeviceToHost));
    // Undecayed: the loaded profile plus every routing decision since start-up (the warm-up counts are cleared).
    // Each layer is capped at kProfileMaxPerLayer so that long use keeps the profile a moving average.
    std::vector<float> fr(freq_.size());
    constexpr double kProfileMaxPerLayer = 1e8;
    for (int l = 0; l < NL; ++l) {
        double sum = 0;
        for (int e = 0; e < c.n_expert; ++e) {
            const size_t i = (size_t)l * c.n_expert + e;
            sum += (double)prior_[i] + route_host_[i];
        }
        const double k = sum > kProfileMaxPerLayer ? kProfileMaxPerLayer / sum : 1.0;
        for (int e = 0; e < c.n_expert; ++e) {
            const size_t i = (size_t)l * c.n_expert + e;
            fr[i] = (float)(((double)prior_[i] + route_host_[i]) * k);
        }
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) { err = "cannot write " + path; return false; }
    const uint32_t hdr[3] = {kProfileMagic, (uint32_t)NL, (uint32_t)c.n_expert};
    f.write((const char*)hdr, sizeof(hdr));
    f.write((const char*)fr.data(), fr.size() * 4);
    return true;
}

// ================================================================================================= decode
void Engine::check_decode_launch(cudaStream_t s, int il, const char* stage) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) die("CUDA launch at step %u layer %d %s: %s", seq_, il, stage, cudaGetErrorString(e));
    if (trace_steps_) {
        log("step %u layer %d: %s queued", seq_, il, stage);
        if (sync_moe_) {
            const cudaError_t sync = cudaStreamSynchronize(s);
            if (sync != cudaSuccess)
                die("CUDA execution at step %u layer %d %s: %s", seq_, il, stage, cudaGetErrorString(sync));
        }
    }
}

void Engine::invalidate_prompt_route_graphs() {
    // Only callers that drained stream_ may destroy these graphs: attention nodes capture KV pointers/stride.
    for (auto& graph : prompt_route_graph_) {
        if (graph) CUDA_CHECK(cudaGraphExecDestroy(graph));
        graph = nullptr;
    }
}

void Engine::capture_prompt_route_graph(int il) {
    const Config& c = model_.cfg;
    SQ_CHECK(prompt_route_graphs_ && prompt_route_capture_stream_ && il >= 0 && il < c.n_layer &&
                 event_moe_ && !event_graph_moe_ && !opt_.use_graph && !profile_pipeline_ && !trace_steps_ &&
                 head_last_only_ && skip_prompt_snapshots_ && !debug_layers,
             "invalid T4 prompt route capture");
    if (prompt_route_graph_[il]) return;
    const LayerW& L = model_.layers[il];
    constexpr int T = 4;
    const float eps = c.eps;
    const cudaStream_t s = prompt_route_capture_stream_;
    const StepParams* P = params_dev_;
    size_t free_before = 0, free_after = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
    const double started = now_ms();
    cudaGraph_t graph = nullptr;
    // The dedicated stream only records kernel nodes, never executes them. This avoids capturing pending
    // compute/copy-stream dependencies. Launch on stream_ later follows the ordinary StepParams copy,
    // embedding/initial norm and preceding layer combine. Every input/state/scratch pointer is engine-owned.
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    if (!L.attn) {
        k_gemv_q8(L.qkvz, xq_, qkvz_, T, s);
        k_gemv_f32(L.ba, xn_, ba_, T, s);
        k_gdn_conv(qkvz_, conv_st_[il], L.conv_w, conv_out_, T, nullptr, state_floats_, s);
        k_gdn_recur(conv_out_, ba_, L.dt_bias, L.ssm_a, ssm_st_[il], o_, T, nullptr, state_floats_, s,
                    gdn_prompt_factors_);
        k_gdn_norm_gate(o_, qkvz_, L.ssm_norm, eps, yq_, T, s);
        k_gemv_q8(L.ssm_out, yq_, attn_out_, T, s);
    } else {
        k_gemv_q8(L.qkv, xq_, qkv_, T, s);
        if (kv_int8_)
            k_attn_prep_i8(qkv_, L.q_norm, L.k_norm, eps, P, c.rope_base, qbuf_, kc_i8_[il], vc_i8_[il],
                           kc_scale_[il], vc_scale_[il], kv_capacity_, T, s);
        else k_attn_prep(qkv_, L.q_norm, L.k_norm, eps, P, c.rope_base, qbuf_, kc_[il], vc_[il], kv_capacity_, T, s);
        if (kv_int8_)
            k_attn_decode_i8(qbuf_, kc_i8_[il], vc_i8_[il], kc_scale_[il], vc_scale_[il], P, kv_capacity_,
                             part_o_, part_ml_, T, s);
        else k_attn_decode(qbuf_, kc_[il], vc_[il], P, kv_capacity_, part_o_, part_ml_, T, s);
        k_attn_combine(part_o_, part_ml_, qkv_, P, kv_capacity_, yq_, T, s);
        k_gemv_q8(L.wo, yq_, attn_out_, T, s);
    }
    k_add_rmsnorm_q(x_, attn_out_, L.post_norm, eps, xn_, xq_, T, s);
    k_gemv_f32(L.router, xn_, rlog_, T, s);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    // Deliberately stop BEFORE the cache-layer readiness wait, router, host mailbox copy or any CPU work.
    // Residency and cache slab replacements therefore need no graph invalidation; KV replacements do.
    size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphInstantiate(&prompt_route_graph_[il], graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    ++prompt_route_captures_;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    log("prompt route graph: layer %d T4 KV capacity %d nodes %zu capture %.2f ms; CUDA free %.2f -> %.2f MiB (delta %.2f MiB), cache %.2f MiB/%lld slots",
        il, kv_capacity_, nodes, now_ms() - started, free_before / 1048576.0, free_after / 1048576.0,
        ((double)free_before - (double)free_after) / 1048576.0, cache_bytes_ / 1048576.0,
        (long long)cache_slots_total_);
}

void Engine::build_layer(cudaStream_t s, int il, int T, const StepParams* P, const float* next_w, float* snap) {
    const Config& c = model_.cfg;
    double cpu_started = 0;
    PipelineLayer* profile = profile_pipeline_ ? &pipeline_layers_[il] : nullptr;
    if (profile) {
        profile->cpu_ms = profile->cpu_wait_ms = profile->router_wait_ms = 0;
        CUDA_CHECK(cudaEventRecord(profile->event[0], s));
    }
    const float eps = c.eps;
    const LayerW& L = model_.layers[il];
    const bool mtp_tail = mtp_last_row_ && il == c.n_layer;
    // Normally these are the complete token arrays. The optional MTP path selects its final row only AFTER
    // every pending pair has written K/V, retaining the original row index for the head and mtp_hlast_.
    float* residual = x_;
    float* normalized = xn_;
    ActQ activation = xq_;
    const bool early_gpu = head_last_only_ && prompt_early_gpu_ != 2 ? prompt_early_gpu_ != 0 : early_gpu_moe_;
    const bool route_graph = prompt_route_graphs_ && il < c.n_layer && T == 4 && head_last_only_ &&
                             skip_prompt_snapshots_ && !debug_layers;
    if (route_graph) {
        SQ_CHECK(event_moe_ && !event_graph_moe_ && !opt_.use_graph && !profile && !trace_steps_ &&
                     P == params_dev_ && snap == nullptr && s == stream_,
                 "invalid prompt-only GPU route graph mode");
        capture_prompt_route_graph(il);
        CUDA_CHECK(cudaGraphLaunch(prompt_route_graph_[il], s));
        ++prompt_route_launches_;
    } else if (!L.attn) {
        float* sc = snap ? snap + (conv_st_[il] - state_arena_) : nullptr;
        float* ss = snap ? snap + (ssm_st_[il] - state_arena_) : nullptr;
        k_gemv_q8(L.qkvz, xq_, qkvz_, T, s);
        check_decode_launch(s, il, "gdn qkvz projection");
        k_gemv_f32(L.ba, xn_, ba_, T, s);
        check_decode_launch(s, il, "gdn beta/alpha projection");
        k_gdn_conv(qkvz_, conv_st_[il], L.conv_w, conv_out_, T, sc, state_floats_, s);
        check_decode_launch(s, il, "gdn convolution");
        k_gdn_recur(conv_out_, ba_, L.dt_bias, L.ssm_a, ssm_st_[il], o_, T, ss, state_floats_, s,
                    head_last_only_ ? gdn_prompt_factors_ : nullptr);
        check_decode_launch(s, il, "gdn recurrence");
        k_gdn_norm_gate(o_, qkvz_, L.ssm_norm, eps, yq_, T, s);
        check_decode_launch(s, il, "gdn norm/gate");
        k_gemv_q8(L.ssm_out, yq_, attn_out_, T, s);
        check_decode_launch(s, il, "gdn output projection");
    } else {
        k_gemv_q8(L.qkv, xq_, qkv_, T, s);
        check_decode_launch(s, il, "attention qkv projection");
        if (kv_int8_)
            k_attn_prep_i8(qkv_, L.q_norm, L.k_norm, eps, P, c.rope_base, qbuf_, kc_i8_[il], vc_i8_[il],
                           kc_scale_[il], vc_scale_[il], kv_capacity_, T, s);
        else k_attn_prep(qkv_, L.q_norm, L.k_norm, eps, P, c.rope_base, qbuf_, kc_[il], vc_[il], kv_capacity_, T, s);
        check_decode_launch(s, il, "attention preparation");
        if (mtp_tail && mtp_state_only_) {
            // This is one full-attention MTP layer: its KV depends only on the input projection/norm.
            // No later layer or future position consumes these discarded attention/MoE outputs.
            if (profile)
                for (int event = 1; event < 5; ++event) CUDA_CHECK(cudaEventRecord(profile->event[event], s));
            ++mtp_kv_only_steps_;
            return;
        }
        const float* attention_q = qbuf_;
        const float* attention_qkv = qkv_;
        if (mtp_tail && T > 1) {
            const int last = T - 1;
            residual += (size_t)last * c.n_embd;
            normalized += (size_t)last * c.n_embd;
            activation = activation.row(last, c.n_embd);
            attention_q += (size_t)last * c.n_head * c.head_dim;
            attention_qkv += (size_t)last * (2 * c.n_head + 2 * c.n_head_kv) * c.head_dim;
            mtp_tail_host_->pos = mparams_host_->pos + last;
            mtp_tail_host_->seq = mparams_host_->seq;
            mtp_tail_host_->n_tok = 1;
            mtp_tail_host_->token[0] = mparams_host_->token[last];
            CUDA_CHECK(cudaMemcpyAsync(mtp_tail_dev_, mtp_tail_host_, step_params_bytes(0), cudaMemcpyHostToDevice, s));
            P = mtp_tail_dev_;
            T = 1;
            ++mtp_last_row_steps_;
        }
        if (kv_int8_)
            k_attn_decode_i8(attention_q, kc_i8_[il], vc_i8_[il], kc_scale_[il], vc_scale_[il], P, kv_capacity_, part_o_, part_ml_, T, s);
        else k_attn_decode(attention_q, kc_[il], vc_[il], P, kv_capacity_, part_o_, part_ml_, T, s);
        check_decode_launch(s, il, "attention decode");
        k_attn_combine(part_o_, part_ml_, attention_qkv, P, kv_capacity_, yq_, T, s);
        check_decode_launch(s, il, "attention combine");
        k_gemv_q8(L.wo, yq_, attn_out_, T, s);
        check_decode_launch(s, il, "attention output projection");
    }
    if (!route_graph) {
        k_add_rmsnorm_q(residual, attn_out_, L.post_norm, eps, normalized, activation, T, s);
        check_decode_launch(s, il, "post-attention norm");
        k_gemv_f32(L.router, normalized, rlog_, T, s);
        check_decode_launch(s, il, "router projection");
    }
    // Cache uploads for later layers can proceed alongside the current layer.  Never route into a slot until
    // both its new blob and this layer's new residency table are visible to this stream.
    // The captured trunk waits for cache_upload_done_ before its launch; avoid capturing a stale/external
    // layer event. Ordinary steps retain their per-layer overlap with ongoing cache uploads.
    if (async_cache_ && !capturing_event_graph_) CUDA_CHECK(cudaStreamWaitEvent(s, cache_layer_ready_[il], 0));
    auto trace_residency = [&](const char* when) {
        std::vector<int32_t> residency((size_t)c.n_expert);
        CUDA_CHECK(cudaMemcpyAsync(residency.data(), residency_dev_ + (size_t)il * c.n_expert,
                                   residency.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        const int resident = (int)std::count_if(residency.begin(), residency.end(), [](int32_t slot) { return slot >= 0; });
        log("step %u layer %d: residency %s router, allocated slots=%d resident=%d first=%d,%d,%d,%d,%d,%d,%d,%d",
            seq_, il, when, slots_per_layer_[il], resident, residency[0], residency[1], residency[2], residency[3],
            residency[4], residency[5], residency[6], residency[7]);
        if (slots_per_layer_[il] == 0)
            SQ_CHECK(std::all_of(residency.begin(), residency.end(), [](int32_t slot) { return slot == -1; }),
                     "empty layer %d has non--1 device residency %s router", il, when);
        return residency;
    };
    std::vector<int32_t> residency_before;
    if (trace_steps_ && sync_moe_) residency_before = trace_residency("before");
    int streamed_here = 0;
    auto begin_cpu_misses = [&] {
        // router_ready_ precedes every GPU expert kernel. Even with early submission, this waits only for the
        // completed mailbox, not the GPU expert result; CPU outputs still enter the stream after end_layer().
        const double router_wait_start = profile ? now_ms() : 0;
        CUDA_CHECK(cudaEventSynchronize(router_ready_));
        if (profile) profile->router_wait_ms = now_ms() - router_wait_start;
        const Mailbox& m = mb_host_[il];
        SQ_CHECK(m.ready_seq == seq_ && m.n_tok == T && m.n_miss >= 0 && m.n_miss <= T * kMaxK,
                 "event MoE invalid mailbox: layer %d ready %u expected %u tokens %d/%d misses %d",
                 il, m.ready_seq, seq_, m.n_tok, T, m.n_miss);
        if (miss_stream_capacity_) streamed_here = prepare_miss_stream(il, T, s);
        if (m.n_miss > 0) {
            if (rebalance_every_ > 0 || profile) cpu_started = now_ms();
            cpu_->begin_layer(il);
        }
    };
    Mailbox* router_mailbox = event_moe_ ? router_mb_dev_ + il : mb_dev_ + il;
    k_router(rlog_, il, residency_dev_, cinfo_dev_, normalized, hits_, router_mailbox, P, d_nmiss_, d_sg_, route_counts_, ecount_, T, s, event_moe_);
    check_decode_launch(s, il, "router");
    if (event_moe_) {
        const size_t mailbox_bytes = offsetof(Mailbox, x) + (size_t)T * sizeof(Mailbox::x[0]);
        CUDA_CHECK(cudaMemcpyAsync(mb_host_ + il, router_mailbox, mailbox_bytes, cudaMemcpyDeviceToHost, s));
        if (capturing_event_graph_) {
            SQ_CHECK(il < c.n_layer, "event graph only supports trunk layers");
            // A real dependency fork: this CPU-only callback may run concurrently with the GPU expert branch.
            // Capture records the callback; it does not execute it or inspect the current mailbox contents.
            CUDA_CHECK(cudaEventRecord(event_graph_fork_, s));
            CUDA_CHECK(cudaStreamWaitEvent(event_graph_cpu_stream_, event_graph_fork_, 0));
            auto* context = capturing_prompt_full_graph_ ? &prompt_full_contexts_[il] :
                &event_graph_contexts_[(size_t)T * c.n_layer + il];
            CUDA_CHECK(cudaLaunchHostFunc(event_graph_cpu_stream_, &Engine::event_graph_cpu, context));
            const size_t result_bytes = offsetof(Result, out) + (size_t)T * sizeof(Result::out[0]);
            CUDA_CHECK(cudaMemcpyAsync(sync_res_dev_, res_host_ + il, result_bytes,
                                       cudaMemcpyHostToDevice, event_graph_cpu_stream_));
            CUDA_CHECK(cudaEventRecord(event_graph_join_, event_graph_cpu_stream_));
        } else {
            CUDA_CHECK(cudaEventRecord(router_ready_, s));
            if (profile) CUDA_CHECK(cudaEventRecord(profile->event[1], s));
            if (!early_gpu) begin_cpu_misses();
        }
    }
    if (trace_steps_ && sync_moe_) {
        const auto residency_after = trace_residency("after");
        SQ_CHECK(residency_before == residency_after, "router changed device residency in layer %d", il);
        HitList h{};
        CUDA_CHECK(cudaMemcpyAsync(&h, hits_, sizeof(h), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        const Mailbox& m = mb_host_[il];
        log("step %u layer %d: routing slots=%d hits=%d pairs=%d misses=%d ready=%u",
            seq_, il, slots_per_layer_[il], h.n, h.n_pair, m.n_miss, m.ready_seq);
        for (int j = 0; j < kMaxK; ++j)
            log("step %u layer %d route %d: CPU id=%d weight=%.7g GPU pointer=%p",
                seq_, il, j, m.ids[j], m.w[j][0], (const void*)h.ptr[j]);
        if (slots_per_layer_[il] == 0)
            SQ_CHECK(h.n == 0 && h.n_pair == 0 && m.n_miss == kMaxK,
                     "zero-cache routing invalid: layer %d hits %d pairs %d misses %d",
                     il, h.n, h.n_pair, m.n_miss);
    }
    if (profile) CUDA_CHECK(cudaEventRecord(profile->event[2], s));
    k_gemv_q8(L.sh_gu, activation, shgu_, T, s);
    check_decode_launch(s, il, "shared expert gate/up");
    k_swiglu_q(shgu_, shq_, c.n_ff_shexp, T, s);
    check_decode_launch(s, il, "shared expert activation");
    k_gemv_q8(L.sh_down, shq_, shout_, T, s);
    check_decode_launch(s, il, "shared expert down");
    if (slots_per_layer_[il] == 0 && !capturing_event_graph_) {
        CUDA_CHECK(cudaMemsetAsync(moe_gpu_, 0, (size_t)T * c.n_embd * sizeof(float), s));
        check_decode_launch(s, il, "zero GPU expert output (empty cache)");
    } else {
        k_moe_gu(L.t_gu, activation, hits_, h_gu_, T, s);
        check_decode_launch(s, il, "GPU experts gate/up");
        k_moe_act(h_gu_, hits_, hq_, T, s);
        check_decode_launch(s, il, "GPU experts activation");
        k_moe_down(L.t_down, 2 * L.gu_bytes, hq_, hits_, moe_gpu_, T, s);
        check_decode_launch(s, il, "GPU experts down");
    }
    if (profile) CUDA_CHECK(cudaEventRecord(profile->event[3], s));
    if (event_moe_ && flush_moe_submit_ && !capturing_event_graph_) {
        // WDDM may hold these launches until the next blocking wait (after CPU
        // misses finish in the original ordering). A query submits the batch
        // without waiting, allowing the intended overlap in either ordering.
        // This is an opt-in, measured driver optimization, not a synchronization
        // dependency: the subsequent result copy/combine remain stream-ordered.
        const cudaError_t submitted = cudaStreamQuery(s);
        if (submitted != cudaSuccess && submitted != cudaErrorNotReady) CUDA_CHECK(submitted);
    }
    // The GPU-only work is independent of CPU misses. Queue it while attention/routing may still execute,
    // then wait for the earlier mailbox event and dispatch the CPU. The optional query above also submits
    // the entire GPU batch on WDDM; the event wait itself never includes these later expert kernels.
    if (event_moe_ && early_gpu && !capturing_event_graph_) begin_cpu_misses();
    if (streamed_here > 0) {
        if (miss_stream_serialized_) {
            // Test reference: transfer completed before CPU dispatch; finish both earlier branches before
            // executing the transient experts. Their inputs, kernels, order and selected IDs are identical.
            if (mb_host_[il].n_miss > 0) cpu_->end_layer(head_last_only_);
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
        CUDA_CHECK(cudaStreamWaitEvent(s, miss_stream_ready_, 0));
        // The same compute stream has finished using resident scratch before it reuses these buffers.
        // The transient bank lives through combine; the next router-ready event proves safe bank reuse.
        // The serialized regression deliberately retains the original full launch grid as its oracle.
        k_moe_gu(L.t_gu, activation, miss_stream_hits_dev_, h_gu_, T, s, miss_stream_serialized_ ? 0 : streamed_here);
        k_moe_act(h_gu_, miss_stream_hits_dev_, hq_, T, s);
        k_moe_down(L.t_down, 2 * L.gu_bytes, hq_, miss_stream_hits_dev_, miss_stream_out_, T, s);
        k_add_expert_output(moe_gpu_, miss_stream_out_, T * c.n_embd, s);
        check_decode_launch(s, il, "transient streamed experts");
        if (profile) CUDA_CHECK(cudaEventRecord(profile->event[3], s)); // include the transient branch/copy wait
        if (flush_moe_submit_) {
            const cudaError_t submitted = cudaStreamQuery(s);
            if (submitted != cudaSuccess && submitted != cudaErrorNotReady) CUDA_CHECK(submitted);
        }
    }
    const Result* result = res_dev_ + il;
    if (capturing_event_graph_) {
        // Both expert branches must finish before combine. No kernel polls a CPU-written mapped flag:
        // the device result is copied only after the host callback returned and the workers completed.
        CUDA_CHECK(cudaStreamWaitEvent(s, event_graph_join_, 0));
        result = sync_res_dev_;
    } else if (event_moe_) {
        // Shared/resident kernels are already running.  Wait only for the CPU job, then order its copied
        // completion flag and all output rows before combine.  The single device result buffer is stream-ordered.
        const Mailbox& m = mb_host_[il];
        if (m.n_miss > 0) {
            const double wait_start = profile ? now_ms() : 0;
            cpu_->end_layer(head_last_only_);
            if (profile) {
                profile->cpu_wait_ms = now_ms() - wait_start;
                profile->cpu_ms = now_ms() - cpu_started;
            }
            if (rebalance_every_ > 0) {
                int pairs = 0;
                for (int j = 0; j < m.n_miss; ++j)
                    for (int t = 0; t < T; ++t) pairs += m.w[j][t] != 0.f;
                const double per_pair = (now_ms() - cpu_started) / std::max(pairs, 1);
                if (std::isfinite(per_pair) && per_pair > 0)
                    cpu_pair_ms_[il] = cpu_pair_ms_[il] > 0 ? 0.9 * cpu_pair_ms_[il] + 0.1 * per_pair : per_pair;
            }
        }
        res_host_[il].done_seq = seq_;
        const size_t result_bytes = offsetof(Result, out) + (size_t)T * sizeof(Result::out[0]);
        CUDA_CHECK(cudaMemcpyAsync(sync_res_dev_, res_host_ + il, result_bytes, cudaMemcpyHostToDevice, s));
        result = sync_res_dev_;
        if (trace_steps_) log("step %u layer %d: event CPU experts complete, tokens %d, misses %d",
                               seq_, il, T, m.n_miss);
    } else if (sync_moe_) {
        SQ_CHECK(T == 1, "STRATA_SYNC_MOE=1 only supports single-token decode");
        const double t0 = now_ms();
        if (trace_steps_) log("step %u layer %d: synchronizing GPU router/shared experts", seq_, il);
        CUDA_CHECK(cudaStreamSynchronize(s));
        const Mailbox& m = mb_host_[il];
        SQ_CHECK(m.ready_seq == seq_ && m.n_tok == 1 && m.n_miss >= 0 && m.n_miss <= kMaxK,
                 "synchronous MoE invalid mailbox: layer %d ready %u expected %u tokens %d misses %d",
                 il, m.ready_seq, seq_, m.n_tok, m.n_miss);
        if (trace_steps_) log("step %u layer %d: router ready, %d CPU experts (GPU %.1f ms)",
                               seq_, il, m.n_miss, now_ms() - t0);
        if (m.n_miss > 0) {
            float weights[kMaxK];
            for (int j = 0; j < m.n_miss; ++j) weights[j] = m.w[j][0];
            cpu_->compute_layer_sync(il, m.x[0], m.ids, weights, m.n_miss, res_host_[il].out[0]);
        }
        res_host_[il].done_seq = seq_;
        // An explicit copy makes both the result and completion flag stream-ordered device data.
        CUDA_CHECK(cudaMemcpyAsync(sync_res_dev_, res_host_ + il, sizeof(Result), cudaMemcpyHostToDevice, s));
        result = sync_res_dev_;
        if (trace_steps_) log("step %u layer %d: CPU experts complete (total %.1f ms)", seq_, il, now_ms() - t0);
    }
    k_combine(residual, moe_gpu_, shout_, d_sg_, d_nmiss_, result, P, next_w, eps, normalized, activation, d_err_, wait_ns_, T, s);
    check_decode_launch(s, il, "CPU/GPU result combine");
    if (profile) CUDA_CHECK(cudaEventRecord(profile->event[4], s));
    if (debug_layers && il < c.n_layer) {
        std::vector<float> h((size_t)c.n_embd * 3);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), x_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(h.data() + c.n_embd, attn_out_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(h.data() + 2 * c.n_embd, moe_gpu_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        debug_layers->push_back(std::move(h));
    }
}

void Engine::build_decode(cudaStream_t s, int T) {
    const Config& c = model_.cfg;
    CUDA_CHECK(cudaMemcpyAsync(params_dev_, params_host_, step_params_bytes(T), cudaMemcpyHostToDevice, s));
    k_embed_in(params_dev_, x_, T, s);
    check_decode_launch(s, -1, "embedding");
    k_rmsnorm_q(x_, model_.layers[0].attn_norm, c.eps, xn_, xq_, T, s);
    check_decode_launch(s, -1, "initial norm");
    // a verification step keeps the delta-net state after each of its tokens, for the rollback
    float* snap = T > 1 && !skip_prompt_snapshots_ ? snap_arena_ : nullptr;
    if (T > 1 && skip_prompt_snapshots_ && !capturing_prompt_full_graph_)
        prompt_snapshot_bytes_skipped_ += (uint64_t)(T - 1) * state_floats_ * sizeof(float);
    for (int il = 0; il < c.n_layer; ++il) {
        const float* next_w = il + 1 < c.n_layer ? model_.layers[il + 1].attn_norm : model_.output_norm;
        build_layer(s, il, T, params_dev_, next_w, snap);
    }
    if (skip_trunk_head_ && !capturing_prompt_full_graph_) ++prompt_heads_skipped_;
    if (!model_.cpu_lm_head && !skip_trunk_head_) {
        // Prompt batches consume only their final logits. Verification keeps every row and its argmax.
        const int first = head_last_only_ ? T - 1 : 0;
        const int nt = head_last_only_ ? 1 : T;
        k_gemv_q8(model_.lm_head, xq_.row(first, c.n_embd), logits_ + (size_t)first * c.n_vocab, nt, s);
        if (T > 1 && !head_last_only_) k_argmax(logits_, c.n_vocab, out_dev_, T, s);
    }
    // the normed final hidden states seed the MTP pairs of these positions
    if (mtp_on_) CUDA_CHECK(cudaMemcpyAsync(mtp_hin_, xn_, (size_t)T * c.n_embd * 4, cudaMemcpyDeviceToDevice, s));
}

void Engine::build_mtp(cudaStream_t s, int T) {
    const Config& c = model_.cfg;
    const MtpW& M = model_.mtp;
    const int il = c.n_layer;
    CUDA_CHECK(cudaMemcpyAsync(mparams_dev_, mparams_host_, step_params_bytes(T), cudaMemcpyHostToDevice, s));
    k_mtp_in(mparams_dev_, mtp_hin_, M.enorm, M.hnorm, c.eps, mtp_act_, T, s);
    k_gemv_q8(M.eh_proj, mtp_act_, x_, T, s);
    k_rmsnorm_q(x_, model_.layers[il].attn_norm, c.eps, xn_, xq_, T, s);
    build_layer(s, il, T, mparams_dev_, M.head_norm, nullptr);
    if (mtp_last_row_ && mtp_state_only_) return; // no consumer needs mtp_hlast_ until a later drafting catch-up
    // only the last pair drafts: its normed hidden state seeds the next (recursive) draft
    CUDA_CHECK(cudaMemcpyAsync(mtp_hlast_, xn_ + (size_t)(T - 1) * c.n_embd, (size_t)c.n_embd * 4, cudaMemcpyDeviceToDevice, s));
    if (!model_.cpu_lm_head && !mtp_state_only_) {
        if (mtp_subset_ && mtp_subset_->active) {
            mtp_subset_->project(model_.lm_head, xq_.row(T - 1, c.n_embd), mtp_logits_, out_dev_ + kMaxT, prob_dev_, s);
        } else {
            k_gemv_q8(model_.lm_head, xq_.row(T - 1, c.n_embd), mtp_logits_, 1, s);
            k_argmax(mtp_logits_, c.n_vocab, out_dev_ + kMaxT, 1, s, prob_dev_);
        }
    }
}

void Engine::run_cpu_lm_head(bool mtp, int T) {
    if (mtp && mtp_state_only_) return;
    if (!mtp && skip_trunk_head_) return;
    const Config& c = model_.cfg;
    const DQ8& W = model_.lm_head;
    const bool last_only = !mtp && head_last_only_;
    const int nt = mtp || last_only ? 1 : T;
    const ActQ a = xq_.row(mtp || last_only ? T - 1 : 0, c.n_embd);
    // Preserve the GPU's activation quantization exactly; only the projection moves to the CPU.
    CUDA_CHECK(cudaMemcpyAsync(head_q_.data(), a.q, (size_t)nt * c.n_embd, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(head_d_.data(), a.d, (size_t)nt * (c.n_embd / 32) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const int workers = cpu_->threads();
    SQ_CHECK(workers > 0, "CPU LM head requires the worker pool");
    auto project = [&](int worker) {
        const int first = c.n_vocab * worker / workers;
        const int last = c.n_vocab * (worker + 1) / workers;
        cpu_lm_head_q8_rows_multi(W.qs, W.d, head_q_.data(), head_d_.data(), head_logits_.data(),
                                  c.n_embd, c.n_vocab, first, last, nt);
    };
    cpu_->parallel_for_workers(project);
    // Prompt batching commits row T-1 to row zero; verification still projects all T rows and samples each.
    float* logits = mtp ? mtp_logits_ : logits_ + (last_only ? (size_t)(T - 1) * c.n_vocab : 0);
    CUDA_CHECK(cudaMemcpyAsync(logits, head_logits_.data(), (size_t)nt * c.n_vocab * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    if (mtp) k_argmax(logits, c.n_vocab, out_dev_ + kMaxT, 1, stream_, prob_dev_);
    else if (T > 1 && !last_only) k_argmax(logits, c.n_vocab, out_dev_, T, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Engine::invalidate_event_graphs() {
    for (auto& graph : event_graph_) {
        if (graph) CUDA_CHECK(cudaGraphExecDestroy(graph));
        graph = nullptr;
    }
}

void Engine::invalidate_prompt_full_graphs() {
    for (auto& graph : prompt_full_graph_exec_) {
        if (graph) CUDA_CHECK(cudaGraphExecDestroy(graph));
        graph = nullptr;
    }
}

void CUDART_CB Engine::event_graph_cpu(void* data) {
    // This callback and the worker functions it invokes must remain CPU-only. In particular, CUDA waits,
    // copies, event records and error queries belong to graph nodes or to the host outside the callback.
    const auto& context = *static_cast<EventGraphContext*>(data);
    Engine& engine = *context.engine;
    const int layer = context.layer;
    const Mailbox& mailbox = engine.mb_host_[layer];
    const uint32_t expected = engine.params_host_->seq;
    SQ_CHECK(mailbox.ready_seq == expected && mailbox.n_tok == context.tokens && mailbox.n_miss >= 0 &&
             mailbox.n_miss <= context.tokens * kMaxK,
             "event graph invalid mailbox: layer %d ready %u expected %u tokens %d/%d misses %d",
             layer, mailbox.ready_seq, expected, mailbox.n_tok, context.tokens, mailbox.n_miss);
    try {
        if (mailbox.n_miss > 0) {
            const double started = engine.rebalance_every_ > 0 ? now_ms() : 0;
            engine.cpu_->begin_layer(layer);
            engine.cpu_->end_layer(context.prompt);
            if (engine.rebalance_every_ > 0) {
                int pairs = 0;
                for (int j = 0; j < mailbox.n_miss; ++j)
                    for (int t = 0; t < context.tokens; ++t) pairs += mailbox.w[j][t] != 0.f;
                const double per_pair = (now_ms() - started) / std::max(pairs, 1);
                if (std::isfinite(per_pair) && per_pair > 0) {
                    double& previous = engine.cpu_pair_ms_[layer];
                    previous = previous > 0 ? 0.9 * previous + 0.1 * per_pair : per_pair;
                }
            }
        }
        engine.res_host_[layer].done_seq = expected;
    } catch (const std::exception& error) {
        die("event graph CPU callback failed at layer %d: %s", layer, error.what());
    } catch (...) {
        die("event graph CPU callback failed at layer %d", layer);
    }
}

void Engine::capture_event_graph(int T) {
    SQ_CHECK(event_graph_moe_ && event_moe_ && !capturing_event_graph_ && !profile_pipeline_ &&
             !head_last_only_ && !debug_layers && cpu_ && T >= 1 && T <= (mtp_on_ ? opt_.mtp_draft + 1 : 1),
             "invalid event-MoE graph capture mode");
    if (event_graph_[T]) return;
    // No outside stream/event dependency is captured. Normal launches wait for the latest completed upload
    // round explicitly; immutable device pointer tables then allow cache contents/capacities to change.
    if (async_cache_) CUDA_CHECK(cudaEventSynchronize(cache_upload_done_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaStreamSynchronize(event_graph_cpu_stream_));
    const double started = now_ms();
    cudaGraph_t graph = nullptr;
    capturing_event_graph_ = true;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    build_decode(stream_, T);
    CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    capturing_event_graph_ = false;
    size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphInstantiate(&event_graph_[T], graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    ++event_graph_captures_;
    log("event MoE trunk graph ready: T=%d, KV capacity=%d, %zu nodes, %.2f ms capture/instantiate",
        T, kv_capacity_, nodes, now_ms() - started);
}

void Engine::capture_prompt_full_graph(bool skip_head) {
    SQ_CHECK(prompt_full_graph_ && event_moe_ && !event_graph_moe_ && !opt_.use_graph &&
                 !prompt_route_graphs_ && !capturing_event_graph_ && !capturing_prompt_full_graph_ &&
                 !profile_pipeline_ && !trace_steps_ && !debug_layers && !miss_stream_capacity_ &&
                 head_last_only_ && skip_prompt_snapshots_ && skip_trunk_head_ == skip_head &&
                 cpu_ && !model_.cpu_lm_head && event_graph_cpu_stream_ && prompt_full_complete_,
             "invalid accepted-T4 prompt full graph capture");
    const int variant = skip_head ? 1 : 0;
    if (prompt_full_graph_exec_[variant]) return;
    // Capture never runs a CPU callback. Drain the preceding (possibly deferred) MTP work first; replay
    // stays on stream_ after that work and reads its own persistent params_host_/params_dev_ buffers.
    if (async_cache_) CUDA_CHECK(cudaEventSynchronize(cache_upload_done_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaStreamSynchronize(event_graph_cpu_stream_));
    size_t free_before = 0, free_after = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
    const double started = now_ms();
    cudaGraph_t graph = nullptr;
    capturing_event_graph_ = true;
    capturing_prompt_full_graph_ = true;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    build_decode(stream_, 4);
    CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    capturing_prompt_full_graph_ = false;
    capturing_event_graph_ = false;
    size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphInstantiate(&prompt_full_graph_exec_[variant], graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    ++prompt_full_captures_;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    log("prompt full graph: %s T4 KV capacity %d nodes %zu capture %.2f ms; CUDA free %.2f -> %.2f MiB "
        "(delta %.2f MiB), cache %.2f MiB/%lld slots",
        skip_head ? "intermediate/no-head" : "final/last-head", kv_capacity_, nodes, now_ms() - started,
        free_before / 1048576.0, free_after / 1048576.0, ((double)free_before - (double)free_after) / 1048576.0,
        cache_bytes_ / 1048576.0, (long long)cache_slots_total_);
}

void Engine::run_step(bool mtp, int T) {
    const Config& c = model_.cfg;
    const double profile_start = profile_pipeline_ ? now_ms() : 0;
    SQ_CHECK(!sync_moe_ || (!mtp && T == 1), "STRATA_SYNC_MOE=1 requires single-token decode with MTP disabled");
    StepParams& P = mtp ? *mparams_host_ : *params_host_;
    ensure_kv_capacity(P.pos + T); // final guard covers trunk verification and every pending/recursive MTP write
    P.seq = ++seq_;
    P.n_tok = T;
    if (!sync_moe_ && !event_moe_) {
        if (mtp) cpu_->begin_step(seq_, c.n_layer, c.n_layer + 1);
        else cpu_->begin_step(seq_, 0, c.n_layer);
    }
    cudaGraphExec_t g = mtp ? mtp_graph_[T] : graph_[T];
    const bool event_graph = event_graph_moe_ && event_graph_ready_ && !mtp && !head_last_only_ && !debug_layers;
    const bool prompt_full = prompt_full_graph_ && !mtp && T == 4 && head_last_only_ &&
                             skip_prompt_snapshots_ && !debug_layers;
    if (prompt_full) {
        const int variant = skip_trunk_head_ ? 1 : 0;
        capture_prompt_full_graph(skip_trunk_head_);
        // Cache slots and their residency are dynamic device data inside the captured router. Wait on the
        // latest upload round OUTSIDE capture; no stale layer-event milestone enters the graph.
        if (async_cache_) CUDA_CHECK(cudaStreamWaitEvent(stream_, cache_upload_done_, 0));
        CUDA_CHECK(cudaGraphLaunch(prompt_full_graph_exec_[variant], stream_));
        ++prompt_full_launches_[variant];
        // These describe executed prompt work, not the one-time construction of the two graph variants.
        if (skip_trunk_head_) ++prompt_heads_skipped_;
        prompt_snapshot_bytes_skipped_ += (uint64_t)(T - 1) * state_floats_ * sizeof(float);
    } else if (event_graph) {
        capture_event_graph(T); // lazily recaptures after KV growth invalidated pointers/strides
        if (async_cache_) CUDA_CHECK(cudaStreamWaitEvent(stream_, cache_upload_done_, 0));
        CUDA_CHECK(cudaGraphLaunch(event_graph_[T], stream_));
        ++event_graph_steps_;
    } else if (g) CUDA_CHECK(cudaGraphLaunch(g, stream_));
    else if (mtp) build_mtp(stream_, T);
    else build_decode(stream_, T);
    if (profile_pipeline_) CUDA_CHECK(cudaEventRecord(pipeline_end_, stream_));
    if (prompt_full) {
        // The event follows every CPU callback/result-copy/GPU join and the optional final head. Keep the
        // main host asleep while the CUDA callback thread dispatches/waits on the ordinary worker pool.
        CUDA_CHECK(cudaEventRecord(prompt_full_complete_, stream_));
        CUDA_CHECK(cudaEventSynchronize(prompt_full_complete_));
    } else if (event_graph && event_graph_blocking_) {
        // Outside capture, after the graph's final GPU/CPU branch join. Sleeping here leaves the main host
        // thread available to the CPU workers; the callback remains responsible for its own worker wait.
        CUDA_CHECK(cudaEventRecord(event_graph_complete_, stream_));
        CUDA_CHECK(cudaEventSynchronize(event_graph_complete_));
    } else CUDA_CHECK(cudaStreamSynchronize(stream_));
    const bool cpu_ok = sync_moe_ || event_moe_ || cpu_->end_step();
    if (!cpu_ok || tok_host_[1] != 0) {
        std::string lay;
        for (int l = 0; l < c.n_moe_layers(); ++l) {
            char b[64];
            std::snprintf(b, sizeof(b), " L%d:r%u/d%u/n%d", l, mb_host_[l].ready_seq, res_host_[l].done_seq,
                          mb_host_[l].n_miss);
            lay += b;
        }
        die("CPU/GPU expert hand-off failed (seq %u, %s step of %d, cpu %s, gpu flag %d):%s", seq_, mtp ? "MTP" : "trunk", T,
            cpu_ok ? "ok" : "timed out", tok_host_[1], lay.c_str());
    }
    if (model_.cpu_lm_head) {
        if (trace_steps_) log("step %u: CPU LM head starting", seq_);
        run_cpu_lm_head(mtp, T);
        if (trace_steps_) log("step %u: CPU LM head complete", seq_);
    }
    if (profile_pipeline_) {
        auto elapsed = [](cudaEvent_t a, cudaEvent_t b) {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
            return (double)ms;
        };
        const int first = mtp ? c.n_layer : 0, last = mtp ? c.n_layer + 1 : c.n_layer;
        double route = 0, launch_gap = 0, expert = 0, combine_gap = 0;
        double cpu = 0, cpu_wait = 0, router_wait = 0;
        for (int layer = first; layer < last; ++layer) {
            const auto& p = pipeline_layers_[layer];
            route += elapsed(p.event[0], p.event[1]);
            launch_gap += elapsed(p.event[1], p.event[2]);
            expert += elapsed(p.event[2], p.event[3]);
            combine_gap += elapsed(p.event[3], p.event[4]);
            cpu += p.cpu_ms; cpu_wait += p.cpu_wait_ms; router_wait += p.router_wait_ms;
        }
        const double head = elapsed(pipeline_layers_[last - 1].event[4], pipeline_end_);
        log("PIPELINE seq=%u phase=%s T=%d pos=%d wall_ms=%.4f route_span_ms=%.4f launch_gap_ms=%.4f expert_span_ms=%.4f combine_gap_ms=%.4f head_tail_ms=%.4f cpu_job_host_ms=%.4f cpu_block_host_ms=%.4f router_block_host_ms=%.4f",
            seq_, mtp ? "mtp" : "trunk", T, P.pos, now_ms() - profile_start,
            route, launch_gap, expert, combine_gap, head, cpu, cpu_wait, router_wait);
    }
    if (!mtp) ++stats.graph_steps;
}

void Engine::decode_one(int token) {
    SQ_CHECK(n_past_ < opt_.ctx, "context full (%d tokens)", opt_.ctx);
    ensure_kv_capacity(n_past_ + 1);
    // the pending MTP pairs end with this token: fill their MTP KV entries before the trunk moves on
    if (mtp_on_ && mtp_k_ > 0) {
        const double tm = now_ms();
        mtp_catch_up(token, false);
        stats.mtp_ms += now_ms() - tm;
    }
    StepParams& P = *params_host_;
    P.pos = n_past_;
    P.token[0] = token;
    model_.embed(token, P.emb[0]);
    const double t0 = now_ms();
    run_step(false, 1);
    const double dt = now_ms() - t0;
    stats.decode_ms += dt;
    if (stats.step_ms.size() < 4096) stats.step_ms.push_back((float)dt);
    else stats.step_ms[stats.decode_steps % 4096] = (float)dt;
    ++stats.decode_steps;
    ++n_past_;
    history_.push_back(token);
    logits_valid_ = !skip_trunk_head_;
    if (mtp_on_) mtp_k_ = 1;
    if (opt_.adapt_every > 0 && cache_slots_total_ > 0 && ++steps_since_adapt_ >= opt_.adapt_every) {
        steps_since_adapt_ = 0;
        const double ta = now_ms();
        adapt_cache();
        stats.adapt_ms += now_ms() - ta;
    }
}

void Engine::decode_multi(const int* tokens, int T) {
    SQ_CHECK(n_past_ + T <= opt_.ctx, "context full (%d tokens)", opt_.ctx);
    ensure_kv_capacity(n_past_ + T);
    StepParams& P = *params_host_;
    P.pos = n_past_;
    for (int t = 0; t < T; ++t) {
        P.token[t] = tokens[t];
        model_.embed(tokens[t], P.emb[t]);
    }
    run_step(false, T);
}

int Engine::mtp_catch_up(int next, bool need_draft, bool defer_completion) {
    const int k = mtp_k_;
    SQ_CHECK(k >= 1 && k <= kMaxT, "MTP: %d pending pairs", k);
    const int q = n_past_ - k;
    StepParams& P = *mparams_host_;
    P.pos = q;
    for (int j = 0; j < k; ++j) {
        // pair j: hidden state of position q + j, token of position q + j + 1
        const int t = q + j + 1 < n_past_ ? history_[q + j + 1] : next;
        P.token[j] = t;
        model_.embed(t, P.emb[j]);
    }
    struct RestoreMtpHead {
        bool& flag;
        bool previous;
        ~RestoreMtpHead() { flag = previous; }
    } restore{mtp_state_only_, mtp_state_only_};
    // Existing graphs contain the complete head and are left unchanged. Uncaptured GPU/CPU paths omit the
    // unused output projection; STRATA_MTP_LAST_ROW additionally stops state-only catch-up after KV writes.
    mtp_state_only_ = !need_draft && !opt_.use_graph;
    if (defer_completion) {
        SQ_CHECK(prompt_async_mtp_ && !need_draft && mtp_state_only_ && mtp_last_row_ && mtp_on_ &&
                     event_moe_ && !event_graph_moe_ && !opt_.use_graph && !profile_pipeline_ && !debug_layers,
                 "deferred MTP catch-up is restricted to uncaptured, GPU-only prompt KV updates");
        ensure_kv_capacity(P.pos + k);
        P.seq = ++seq_;
        P.n_tok = k;
        // LAST_ROW + state-only returns from build_mtp immediately after the single MTP layer writes K/V:
        // no routing, CPU workers, head or host result is consumed. The next trunk shares stream_, so its
        // embedding/scratch writes and final mtp_hin_ copy follow these reads. Its router/completion waits
        // drain this work before feed can rewrite mparams_host_ or enqueue another catch-up. Lazy KV growth
        // also drains the stream before replacing any K/V allocation. Do not use this branch outside feed.
        build_mtp(stream_, k);
        ++prompt_mtp_submitted_;
    } else run_step(true, k);
    mtp_k_ = 0;
    return need_draft ? out_host_[kMaxT] : -1;
}

int Engine::mtp_recur(int prev_draft, int pos) {
    CUDA_CHECK(cudaMemcpyAsync(mtp_hin_, mtp_hlast_, (size_t)model_.cfg.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
    StepParams& P = *mparams_host_;
    P.pos = pos;
    P.token[0] = prev_draft;
    model_.embed(prev_draft, P.emb[0]);
    run_step(true, 1);
    return out_host_[kMaxT];
}

void Engine::spec_step(int tok, const SamplingParams& sp, std::mt19937_64& rng, std::vector<int>& out,
                       const std::vector<int>* stop_tokens, int max_output) {
    SQ_CHECK(max_output > 0, "spec_step: max_output must be positive");
    out.clear();
    const Config& c = model_.cfg;
    int D = std::min({mtp_draft(), opt_.ctx - n_past_ - 1, max_output - 1});
    if (D <= 0 || mtp_k_ <= 0) {
        feed(&tok, 1);
        out.push_back(sample(sp, rng));
        return;
    }
    ensure_kv_capacity(n_past_ + D + 1); // reserve the complete possible draft/verify window before writing any of it
    const double t0 = now_ms();
    // Fix the subset for this round using the last committed target logits and actual recent token IDs. The
    // proposal is still one deterministic argmax token: sample_speculative's point-mass correction is unchanged.
    struct SubsetScope {
        MtpDraftSubset* subset;
        ~SubsetScope() { if (subset) subset->active = false; }
    } subset_scope{mtp_subset_.get()};
    if (mtp_subset_) {
        mtp_subset_->prepare(logits_, history_, tok, stream_);
        mtp_subset_->active = true;
    }
    // ---- draft: the pending pairs (ending with tok) give the first draft, the MTP's own hidden state the others.
    // Drafting stops once the MTP's own probability that all drafts so far are right drops below draft_p: a draft
    // that is likely rejected costs a verification slot (and its CPU experts) for nothing.
    int toks[kMaxT];
    toks[0] = tok;
    toks[1] = mtp_catch_up(tok);
    double conf = *prob_host_;
    int nd = 1;
    if (conf < 0.5 * opt_.draft_p) nd = 0;   // not even the first one: a plain step
    for (int i = 2; i <= D && nd == i - 1 && conf >= opt_.draft_p; ++i) {
        toks[i] = mtp_recur(toks[i - 1], n_past_ + i - 2);
        conf *= *prob_host_;
        nd = i;
    }
    const double t1 = now_ms();
    if (mtp_subset_) mtp_subset_->active = false;
    stats.mtp_ms += t1 - t0;
    if (nd == 0) {   // the catch-up is done (mtp_k_ = 0): this feeds tok alone
        ++stats.spec_skipped;
        feed(&tok, 1);
        out.push_back(sample(sp, rng));
        return;
    }
    D = nd;
    // ---- verify [tok, drafts] in one trunk step
    const int T = D + 1;
    decode_multi(toks, T);
    stats.verify_ms += now_ms() - t1;
    // ---- accept: row i of the logits predicts the token after toks[i]
    int a = 0;         // tokens of the step kept (tok + accepted drafts)
    int bonus = -1;    // the trunk's own token after them
    const bool greedy = sp.greedy();
    for (int i = 0; i < T; ++i) {
        a = i + 1;
        int t;
        bool acc = false;
        if (greedy) {
            t = out_host_[i];
            acc = i < D && t == toks[i + 1];
        } else {
            SamplePen pen{sp.presence_penalty, sp.frequency_penalty, sp.repetition_penalty};
            k_candidates(logits_ + (size_t)i * c.n_vocab, c.n_vocab, tok_counts_, pen, cand_tmp_, cand_dev_, stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            t = i < D ? sample_speculative(*cand_host_, sp, rng, toks[i + 1], acc) : sample_candidates(*cand_host_, sp, rng);
            // count it for the penalties of the following rows (in stream order before their candidates; the id
            // is read from mapped memory, which is rewritten only after the next synchronisation)
            count_sampled(t);
        }
        // The final emitted token stays unfed, exactly as in plain decoding.  Even a matching draft must become
        // the bonus at EOS/length: committing it (or later drafts) would retain an un-emitted suffix in history,
        // defeating live-prefix reuse on the next chat turn.  Count only the sampled tokens above for penalties.
        const bool stop = stop_tokens && std::find(stop_tokens->begin(), stop_tokens->end(), t) != stop_tokens->end();
        if (!acc || stop || a == max_output) { bonus = t; break; }
    }
    commit_step(toks, a, T);
    for (int i = 1; i < a; ++i) out.push_back(toks[i]);
    out.push_back(bonus);

    const double dt = now_ms() - t0;
    stats.decode_ms += dt;
    for (int i = 0; i < a; ++i) {
        const float per = (float)(dt / a);
        if (stats.step_ms.size() < 4096) stats.step_ms.push_back(per);
        else stats.step_ms[(stats.decode_steps + i) % 4096] = per;
    }
    stats.decode_steps += a;
    ++stats.spec_steps;
    stats.drafted += D;
    stats.accepted += a - 1;
    steps_since_adapt_ += a;
    if (opt_.adapt_every > 0 && cache_slots_total_ > 0 && steps_since_adapt_ >= opt_.adapt_every) {
        steps_since_adapt_ = 0;
        const double ta = now_ms();
        adapt_cache();
        stats.adapt_ms += now_ms() - ta;
    }
}

void Engine::commit_step(const int* toks, int a, int T) {
    const Config& c = model_.cfg;
    SQ_CHECK(!skip_prompt_snapshots_ || a == T, "prompt snapshot elision requires full acceptance");
    // roll the delta net back to the last kept token; logits row 0 = the last kept token's
    if (a < T)
        CUDA_CHECK(cudaMemcpyAsync(state_arena_, snap_arena_ + (size_t)(a - 1) * state_floats_, state_floats_ * 4,
                                   cudaMemcpyDeviceToDevice, stream_));
    if (a > 1 && !skip_trunk_head_)
        CUDA_CHECK(cudaMemcpyAsync(logits_, logits_ + (size_t)(a - 1) * c.n_vocab, (size_t)c.n_vocab * 4,
                                   cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    n_past_ += a;
    history_.insert(history_.end(), toks, toks + a);
    logits_valid_ = !skip_trunk_head_;
    mtp_k_ = a;   // the step copied the kept tokens' hidden states to mtp_hin_[0 .. a)
}

std::vector<float> Engine::debug_verify(const int* tokens, int T, int keep) {
    SQ_CHECK(mtp_on_ && T >= 2 && T <= opt_.mtp_draft + 1 && keep >= 1 && keep <= T, "debug_verify: bad arguments");
    const size_t V = model_.cfg.n_vocab;
    if (mtp_k_ > 0) mtp_catch_up(tokens[0], false);
    decode_multi(tokens, T);
    std::vector<float> h(V * T);
    CUDA_CHECK(cudaMemcpy(h.data(), logits_, h.size() * 4, cudaMemcpyDeviceToHost));
    commit_step(tokens, keep, T);
    return h;
}

void Engine::feed(const int* tokens, int n) {
    if (n <= 0) return;
    SQ_CHECK(n_past_ + n <= opt_.ctx, "prompt exceeds the context (%d + %d > %d)", n_past_, n, opt_.ctx);
    struct PromptScope {
        bool& head;
        bool& snapshots;
        bool previous_head, previous_snapshots;
        PromptScope(bool& h, bool& s, bool skip_head, bool skip_snapshots)
            : head(h), snapshots(s), previous_head(h), previous_snapshots(s) {
            head = skip_head;
            snapshots = skip_snapshots;
        }
        ~PromptScope() { head = previous_head; snapshots = previous_snapshots; }
    };
    if (prefill_phase_switch_ && opt_.prefill_chunk > 0 && n >= opt_.prefill_min) {
        const double t0 = now_ms();
        // Grow only to this request's end. Lazy KV first returns the necessary cache slots; the remaining slots
        // are then lent to the prompt workspace and restored after the prompt has completed.
        ensure_kv_capacity(n_past_ + n);
        const std::vector<std::vector<int>> previous_layout = slot_expert_;
        const int64_t lent_bytes = cache_bytes_;
        if (lent_bytes > 0) trim_cache(lent_bytes);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
        int64_t snapshot_lent = 0;
        if (prefill_snapshot_loan_ && snap_arena_) {
            snapshot_lent = (size_t)opt_.mtp_draft * state_floats_ * sizeof(float);
            CUDA_CHECK(cudaFree(snap_arena_));
            snap_arena_ = nullptr;
        }
        const int64_t head_lent = model_.release_lm_head();
        log("phase-switched prefill: borrowed %.1f MiB cache + %.1f MiB rollback + %.1f MiB LM head "
            "for %d tokens (chunk %d)", lent_bytes / 1048576.0, snapshot_lent / 1048576.0,
            head_lent / 1048576.0, n, opt_.prefill_chunk);
        prefill_ = std::make_unique<Prefill>(*this);
        std::string prefill_error;
        SQ_CHECK(prefill_->init(prefill_error), "phase-switched prefill initialization failed: %s", prefill_error.c_str());
        prefill_->run(tokens, n);
        prefill_.reset();
        if (head_lent > 0) {
            model_.restore_lm_head();
            k_gemv_q8(model_.lm_head, xq_, logits_, 1, stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        if (snapshot_lent > 0) {
            CUDA_CHECK(cudaMalloc(&snap_arena_, (size_t)snapshot_lent));
        }
        refill_cache_after_prefill(previous_layout);
        stats.prefill_ms += now_ms() - t0;
        stats.prefill_tokens += n;
        n_past_ += n;
        history_.insert(history_.end(), tokens, tokens + n);
        logits_valid_ = true;
        return;
    }
    // Bounded prompt batching reuses the already allocated decode scratch.  Unlike the large prefill path it
    // needs no GEMM/staging workspace.  Keep it opt-in and on the event hand-off until the other paths are audited.
    int prompt_batch = 1;
    if (const char* v = std::getenv("STRATA_PROMPT_BATCH")) {
        SQ_CHECK(std::strcmp(v, "1") == 0 || std::strcmp(v, "2") == 0 || std::strcmp(v, "4") == 0,
                 "STRATA_PROMPT_BATCH must be 1, 2 or 4");
        prompt_batch = v[0] - '0';
    }
    if (prompt_batch > 1) {
        SQ_CHECK(event_moe_ && !sync_moe_,
                 "STRATA_PROMPT_BATCH>1 requires STRATA_EVENT_MOE=1 and STRATA_SYNC_MOE=0");
        // An MTP engine allocates rollback snapshots for its configured verification window (drafts + 1).
        // Plain prompt batches accept every token; still stay within that allocation used by build_decode.
        const int batch_limit = mtp_on_ ? std::min(prompt_batch, opt_.mtp_draft + 1) : prompt_batch;
        for (int i = 0; i < n;) {
            int T = std::min(batch_limit, n - i);
            // Preserve adaptation's token cadence: a batch must not cross a boundary where T=1 would swap.
            if (opt_.adapt_every > 0 && cache_slots_total_ > 0)
                T = std::min(T, std::max(1, opt_.adapt_every - steps_since_adapt_));
            // Intermediate logits have no consumer. Preserve final hidden states and all MTP KV catch-up.
            // Prompt chunks are fully accepted, so verification rollback frames also have no consumer here.
            PromptScope prompt_scope(skip_trunk_head_, skip_prompt_snapshots_,
                                     prompt_skip_head_ && i + T < n, prompt_no_snapshots_);
            if (T == 1) {
                if (prompt_async_mtp_ && !debug_layers && mtp_on_ && mtp_k_ > 0) {
                    const double tm = now_ms();
                    mtp_catch_up(tokens[i], false, true);
                    stats.mtp_ms += now_ms() - tm; // submission only; decode_one drains the queued GPU work
                }
                decode_one(tokens[i++]);
                continue;
            }
            if (mtp_on_ && mtp_k_ > 0) {
                const double tm = now_ms();
                mtp_catch_up(tokens[i], false, prompt_async_mtp_ && !debug_layers);
                stats.mtp_ms += now_ms() - tm;
            }
            const double t0 = now_ms();
            {
                struct RestoreHeadMode {
                    bool& flag;
                    bool previous;
                    ~RestoreHeadMode() { flag = previous; }
                } restore{head_last_only_, head_last_only_};
                head_last_only_ = true;
                decode_multi(tokens + i, T);
            }
            // All prompt tokens are accepted: the recurrent state already holds the last one, and commit moves
            // that token's logits into row zero and appends exactly T tokens to history/n_past (no rollback).
            commit_step(tokens + i, T, T);
            if (!mtp_on_) mtp_k_ = 0; // with MTP, retain all T hidden pairs for the next chunk or first draft
            const double dt = now_ms() - t0;
            stats.decode_ms += dt;
            for (int t = 0; t < T; ++t) {
                const float per_token = (float)(dt / T);
                if (stats.step_ms.size() < 4096) stats.step_ms.push_back(per_token);
                else stats.step_ms[(stats.decode_steps + t) % 4096] = per_token;
            }
            stats.decode_steps += T;
            if (opt_.adapt_every > 0 && cache_slots_total_ > 0) {
                steps_since_adapt_ += T;
                if (steps_since_adapt_ >= opt_.adapt_every) {
                    steps_since_adapt_ = 0;
                    const double ta = now_ms();
                    adapt_cache();
                    stats.adapt_ms += now_ms() - ta;
                }
            }
            i += T;
        }
        return;
    }
    if (n >= opt_.prefill_min && prefill_ && prefill_->ready()) {
        const double t0 = now_ms();
        prefill_->run(tokens, n);   // also runs the MTP layer over the new positions (and leaves one pair pending)
        stats.prefill_ms += now_ms() - t0;
        stats.prefill_tokens += n;
        n_past_ += n;
        history_.insert(history_.end(), tokens, tokens + n);
        logits_valid_ = true;
        if (opt_.adapt_every > 0 && cache_slots_total_ > 0) adapt_cache();
        return;
    }
    for (int i = 0; i < n; ++i) {
        PromptScope prompt_scope(skip_trunk_head_, skip_prompt_snapshots_,
                                 prompt_skip_head_ && i + 1 < n, prompt_no_snapshots_);
        if (prompt_async_mtp_ && !debug_layers && mtp_on_ && mtp_k_ > 0) {
            const double tm = now_ms();
            mtp_catch_up(tokens[i], false, true);
            stats.mtp_ms += now_ms() - tm; // submission only; the immediately following trunk is the completion fence
        }
        decode_one(tokens[i]);
    }
}

void Engine::reset() {
    CUDA_CHECK(cudaMemsetAsync(state_arena_, 0, state_floats_ * 4, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    n_past_ = 0;
    history_.clear();
    logits_valid_ = false;
    mtp_k_ = 0;
}

void Engine::begin_request(const SamplingParams& sp) {
    pen_n_ = sp.penalized() ? (sp.penalty_last_n > 0 ? sp.penalty_last_n : opt_.ctx) : 0;
    pen_hist_.clear();
    if (pen_n_ == 0) return;
    const int from = std::max(0, (int)history_.size() - pen_n_);
    pen_hist_.assign(history_.begin() + from, history_.end());
    std::vector<uint32_t> c((size_t)model_.cfg.n_vocab, 0);
    for (int t : pen_hist_) ++c[(size_t)t];
    CUDA_CHECK(cudaMemcpyAsync(tok_counts_, c.data(), c.size() * 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Engine::count_sampled(int tok) {
    if (pen_n_ == 0) return;
    pen_hist_.push_back(tok);
    const int n = (int)pen_hist_.size();
    tok_host_[2] = tok;
    tok_host_[3] = n > pen_n_ ? pen_hist_[(size_t)(n - 1 - pen_n_)] : -1;
    k_count_token(tok_counts_, tok_dev_ + 2, stream_);
}

void Engine::expert_counts(uint64_t& resident, uint64_t& missing) const {
    unsigned long long ec[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(ec, ecount_, 16, cudaMemcpyDeviceToHost));
    resident = ec[0];
    missing = ec[1];
}

int Engine::sample(const SamplingParams& sp, std::mt19937_64& rng) {
    SQ_CHECK(logits_valid_, "sample() without logits");
    int tok;
    if (sp.greedy()) {
        k_argmax(logits_, model_.cfg.n_vocab, tok_dev_, 1, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        tok = tok_host_[0];
    } else {
        SamplePen pen{sp.presence_penalty, sp.frequency_penalty, sp.repetition_penalty};
        k_candidates(logits_, model_.cfg.n_vocab, tok_counts_, pen, cand_tmp_, cand_dev_, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        tok = sample_candidates(*cand_host_, sp, rng);
    }
    // count it for the penalties: the kernel reads the id from mapped host memory
    count_sampled(tok);
    return tok;
}

// ================================================================================================= prefix reuse
static bool is_prefix(const std::vector<int>& p, const std::vector<int>& s) {
    return p.size() <= s.size() && std::equal(p.begin(), p.end(), s.begin());
}

int Engine::reuse_prefix(const std::vector<int>& prompt) {
    // Positions past the common prefix are about to be rewritten (KV caches), so a checkpoint that is not a prefix
    // of this prompt can no longer be restored.
    for (auto& ck : ckpts_)
        if (!ck.tokens.empty() && !is_prefix(ck.tokens, prompt)) ck.tokens.clear();
    // the live state works if it is a prefix (and, when it covers the whole prompt, still has its logits)
    int best = 0;
    if (!history_.empty() && is_prefix(history_, prompt) && (history_.size() < prompt.size() || logits_valid_))
        best = (int)history_.size();
    Checkpoint* pick = nullptr;
    for (auto& ck : ckpts_)
        if (!ck.tokens.empty() && (int)ck.tokens.size() > best && ck.tokens.size() < prompt.size()) {
            best = (int)ck.tokens.size();
            pick = &ck;
        }
    if (!pick) {
        if (best > 0) return best;
        reset();
        return 0;
    }
    CUDA_CHECK(cudaMemcpyAsync(state_arena_, pick->arena, state_floats_ * 4, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(mtp_hin_, pick->hin, (size_t)kMaxT * model_.cfg.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    pick->stamp = ++ckpt_clock_;
    history_ = pick->tokens;
    n_past_ = (int)history_.size();
    logits_valid_ = false;
    mtp_k_ = pick->mtp_k;
    return n_past_;
}

void Engine::save_checkpoint(bool base) {
    if (ckpts_.empty()) return;
    // slot 0 is the base slot when there are two or more; the others rotate (least recently used)
    const size_t first = ckpts_.size() > 1 && !base ? 1 : 0;
    for (auto& ck : ckpts_)
        if (ck.tokens == history_) { ck.stamp = ++ckpt_clock_; return; }
    Checkpoint* slot = &ckpts_[first];
    if (!base || ckpts_.size() == 1)
        for (size_t i = first; i < ckpts_.size(); ++i)
            if (ckpts_[i].stamp < slot->stamp) slot = &ckpts_[i];
    CUDA_CHECK(cudaMemcpyAsync(slot->arena, state_arena_, state_floats_ * 4, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(slot->hin, mtp_hin_, (size_t)kMaxT * model_.cfg.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    slot->tokens = history_;
    slot->mtp_k = mtp_k_;
    slot->stamp = ++ckpt_clock_;
}

std::string Engine::status_line() const {
    char buf[640];
    unsigned long long wait = 0, ec[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(&wait, wait_ns_, 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ec, ecount_, 16, cudaMemcpyDeviceToHost));
    const double hr = ec[0] + ec[1] ? 100.0 * ec[0] / (ec[0] + ec[1]) : 0.0;
    const double toks = std::max<double>((double)stats.decode_steps, 1.0);
    std::vector<float> st = stats.step_ms;
    double median = 0;
    if (!st.empty()) {
        std::nth_element(st.begin(), st.begin() + st.size() / 2, st.end());
        median = st[st.size() / 2];
    }
    int n = std::snprintf(buf, sizeof(buf),
                          "decode %.1f tok/s (median %.1f), prefill %.0f tok/s, cache hit %.1f%%, swaps %llu "
                          "(per token: %.2f ms, GPU waits for CPU %.2f ms, cache adapt %.3f ms)",
                          stats.decode_ms > 0 ? stats.decode_steps * 1000.0 / stats.decode_ms : 0.0,
                          median > 0 ? 1000.0 / median : 0.0,
                          stats.prefill_ms > 0 ? stats.prefill_tokens * 1000.0 / stats.prefill_ms : 0.0, hr,
                          (unsigned long long)stats.swaps, stats.decode_ms / toks, wait * 1e-6 / toks, stats.adapt_ms / toks);
    if (stats.spec_steps > 0 && n > 0 && n < (int)sizeof(buf))
        std::snprintf(buf + n, sizeof(buf) - n, "; MTP: %.1f%% of drafts accepted, %.2f tokens per step (%.1f%% of steps without a draft), "
                      "drafting %.2f ms + verifying %.2f ms per step",
                      stats.drafted ? 100.0 * stats.accepted / stats.drafted : 0.0,
                      1.0 + (double)stats.accepted / stats.spec_steps,
                      100.0 * stats.spec_skipped / (stats.spec_steps + stats.spec_skipped),
                      stats.mtp_ms / (stats.spec_steps + stats.spec_skipped), stats.verify_ms / stats.spec_steps);
    std::string result(buf);
    if (miss_stream_capacity_) {
        char transient[288];
        std::snprintf(transient, sizeof(transient), "; transient misses: %llu experts, %llu token pairs, %.2f MiB DMA (not cache hits), max unique/pairs %llu/%llu, zero-CPU layers %llu",
                      (unsigned long long)stats.streamed_experts, (unsigned long long)stats.streamed_pairs,
                      stats.streamed_bytes / 1048576.0, (unsigned long long)stats.stream_max_unique,
                      (unsigned long long)stats.stream_max_pairs, (unsigned long long)stats.stream_zero_cpu_layers);
        result += transient;
    }
    return result;
}

void Engine::debug_set_miss_stream_eligible(const std::vector<std::vector<int>>& ids) {
    SQ_CHECK(miss_stream_capacity_ > 0 && ids.size() == model_.layers.size(), "invalid transient eligibility override");
    CUDA_CHECK(cudaStreamSynchronize(miss_copy_stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (size_t layer = 0; layer < ids.size(); ++layer) {
        SQ_CHECK(ids[layer].size() <= (size_t)miss_stream_eligible_limit_, "transient eligibility exceeds configured fixed-ID limit");
        SQ_CHECK(ids[layer].empty() || model_.layers[layer].experts_pinned, "transient source must be explicitly pinned");
        auto sorted = ids[layer];
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 0; i < sorted.size(); ++i)
            SQ_CHECK(sorted[i] >= 0 && sorted[i] < model_.cfg.n_expert && (!i || sorted[i] != sorted[i - 1]),
                     "invalid/duplicate transient expert ID");
        miss_stream_eligible_[layer] = std::move(sorted);
    }
}

void Engine::debug_set_miss_stream_serialized(bool enabled) {
    SQ_CHECK(miss_stream_capacity_ > 0, "transient streaming was not initialized");
    CUDA_CHECK(cudaStreamSynchronize(miss_copy_stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    miss_stream_serialized_ = enabled;
}

std::vector<int> Engine::debug_cpu_miss_ids(int layer) {
    SQ_CHECK(event_moe_ && layer >= 0 && layer < model_.cfg.n_moe_layers(), "invalid mailbox debug layer");
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const Mailbox& mailbox = mb_host_[layer];
    return std::vector<int>(mailbox.ids, mailbox.ids + mailbox.n_miss);
}

bool Engine::debug_miss_stream_payloads_match() {
    if (miss_stream_last_layer_ < 0 || miss_stream_last_ids_.empty()) return false;
    CUDA_CHECK(cudaStreamSynchronize(miss_copy_stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const LayerW& weights = model_.layers[miss_stream_last_layer_];
    std::vector<uint8_t> bytes((size_t)weights.blob_bytes);
    for (size_t slot = 0; slot < miss_stream_last_ids_.size(); ++slot) {
        CUDA_CHECK(cudaMemcpy(bytes.data(), miss_stream_bank_ + slot * miss_stream_stride_, bytes.size(), cudaMemcpyDeviceToHost));
        if (std::memcmp(bytes.data(), weights.expert_blob(miss_stream_last_ids_[slot]), bytes.size()) != 0) return false;
    }
    return true;
}

void Engine::debug_set_mtp_last_row(bool enabled) {
    SQ_CHECK(!enabled || (mtp_tail_host_ && mtp_tail_dev_ && event_moe_ && mtp_on_ && !opt_.use_graph),
             "initialize with STRATA_MTP_LAST_ROW=1 before enabling its debug comparison");
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    mtp_last_row_ = enabled;
}

std::vector<float> Engine::debug_mtp_catch_up(int next, bool need_draft) {
    SQ_CHECK(mtp_on_ && (!mtp_subset_ || !mtp_subset_->active), "debug MTP catch-up requires an inactive draft subset");
    mtp_catch_up(next, need_draft);
    if (!need_draft) return {};
    std::vector<float> logits(model_.cfg.n_vocab);
    CUDA_CHECK(cudaMemcpy(logits.data(), mtp_logits_, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return logits;
}

bool Engine::debug_cache_payloads_match() {
    if (async_cache_) CUDA_CHECK(cudaEventSynchronize(cache_upload_done_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::vector<uint8_t> payload;
    for (int layer = 0; layer < model_.cfg.n_moe_layers(); ++layer) {
        const LayerW& weights = model_.layers[layer];
        payload.resize((size_t)weights.blob_bytes);
        for (int slot = 0; slot < slots_per_layer_[layer]; ++slot) {
            CUDA_CHECK(cudaMemcpy(payload.data(), slot_base_[layer] + (int64_t)slot * weights.blob_bytes,
                                  payload.size(), cudaMemcpyDeviceToHost));
            if (std::memcmp(payload.data(), weights.expert_blob(slot_expert_[layer][slot]), payload.size()) != 0)
                return false;
        }
    }
    return true;
}

std::vector<uint8_t> Engine::debug_prefix_state() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const Config& c = model_.cfg;
    std::vector<uint8_t> result;
    const uint32_t header[] = {(uint32_t)n_past_, (uint32_t)mtp_k_, kv_int8_ ? 1u : 0u};
    result.insert(result.end(), (const uint8_t*)header, (const uint8_t*)header + sizeof(header));
    auto append = [&](const void* source, size_t bytes) {
        if (!bytes) return;
        const size_t offset = result.size();
        result.resize(offset + bytes);
        CUDA_CHECK(cudaMemcpy(result.data() + offset, source, bytes, cudaMemcpyDeviceToHost));
    };
    append(state_arena_, (size_t)state_floats_ * sizeof(float));
    if (mtp_on_ && mtp_k_ > 0) append(mtp_hin_, (size_t)mtp_k_ * c.n_embd * sizeof(float));
    // Only live causal positions are meaningful. Pending MTP pairs have not written their KV positions yet;
    // verification/checkpoint suffixes beyond this prefix must not enter its state identity.
    auto append_heads = [&](const void* source, int positions, size_t token_bytes) {
        if (!source || positions <= 0) return;
        const size_t width = (size_t)positions * token_bytes;
        const size_t offset = result.size();
        result.resize(offset + width * c.n_head_kv);
        CUDA_CHECK(cudaMemcpy2D(result.data() + offset, width, source, (size_t)kv_capacity_ * token_bytes,
                                width, c.n_head_kv, cudaMemcpyDeviceToHost));
    };
    for (int layer = 0; layer < c.n_moe_layers(); ++layer) {
        const int positions = layer < c.n_layer ? n_past_ : std::max(0, n_past_ - mtp_k_);
        if (kv_int8_) {
            append_heads(kc_i8_[layer], positions, c.head_dim * sizeof(int8_t));
            append_heads(vc_i8_[layer], positions, c.head_dim * sizeof(int8_t));
            append_heads(kc_scale_[layer], positions, sizeof(float));
            append_heads(vc_scale_[layer], positions, sizeof(float));
        } else {
            append_heads(kc_[layer], positions, c.head_dim * sizeof(uint16_t));
            append_heads(vc_[layer], positions, c.head_dim * sizeof(uint16_t));
        }
    }
    return result;
}

std::vector<float> Engine::debug_hidden() {
    std::vector<float> h(model_.cfg.n_embd);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaMemcpy(h.data(), x_, h.size() * 4, cudaMemcpyDeviceToHost));
    return h;
}

std::vector<float> Engine::logits_host() {
    std::vector<float> h(model_.cfg.n_vocab);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaMemcpy(h.data(), logits_, h.size() * 4, cudaMemcpyDeviceToHost));
    return h;
}

}  // namespace sq
