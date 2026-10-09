// kernels.cuh - host-callable launchers for the decode (one token) kernels.  All shapes are this model's.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cpu_moe.hpp"
#include "model.hpp"

#ifdef __CUDACC__
#define SQ_HD __host__ __device__
#else
#define SQ_HD
#endif

namespace sq {

/// Activation vectors quantized for the GPU dot products: int8 per value, and per 32-block the scale `d` and
/// `s = d * sum(q)` (the latter is what the Q4_K/Q5_K "min" terms need).  A step of T tokens stores T vectors of n
/// values back to back (row(t, n) addresses token t).
struct ActQ {
    int8_t* q = nullptr;
    float* d = nullptr;
    float* s = nullptr;
    SQ_HD ActQ row(int t, int n) const { return ActQ{q + (size_t)t * n, d + (size_t)t * n / 32, s + (size_t)t * n / 32}; }
};

/// Written by the host before every step (pinned), copied to the device as the step's first node.  A step feeds
/// n_tok tokens at positions pos .. pos + n_tok - 1.
struct alignas(16) StepParams {
    int32_t pos;       // position of the first token
    uint32_t seq;      // step sequence number (mailbox handshake)
    int32_t n_tok;
    int32_t _pad;
    int32_t token[kMaxT];
    float emb[kMaxT][2048];   // the tokens' embedding rows (only the first n_tok are copied)
};
constexpr size_t step_params_bytes(int n_tok) { return offsetof(StepParams, emb) + (size_t)n_tok * 2048 * sizeof(float); }

/// The resident experts of one layer this step, computed by the GPU: distinct experts, the routing weight of each
/// token (0 = not routed), and the index of each (expert, token) pair in the pair-major scratch buffers.
struct HitList {
    int32_t n;
    int32_t n_pair;
    const uint8_t* ptr[kMaxU];
    float w[kMaxU][kMaxT];
    int8_t pair[kMaxU][kMaxT];
};

struct CacheLayerInfo {
    uint8_t* base;       // device slot arena of this layer
    long long blob;      // bytes per slot
};

/// Sampling candidates written to mapped host memory.
constexpr int kCand = 64;
struct Candidates {
    int32_t id[kCand];
    float logit[kCand];
};

struct SamplePen {
    float presence = 0.0f, frequency = 0.0f, repetition = 1.0f;
};

// All launchers take the step's token count T (1..kMaxT, fixed when a graph is captured); vectors of a step are stored
// token-major (token t of an n-vector at t * n).

// ---- elementwise / norms
void k_embed_in(const StepParams* P, float* x, int T, cudaStream_t s);
/// out = rmsnorm(x) * w per token (fp32 optional) and its quantized copy.
void k_rmsnorm_q(const float* x, const float* w, float eps, float* out_f, ActQ out_q, int T, cudaStream_t s);
/// x += a; then xn = rmsnorm(x) * w (fp32) and its quantized copy.
void k_add_rmsnorm_q(float* x, const float* a, const float* w, float eps, float* xn, ActQ xq, int T, cudaStream_t s);
/// MTP input: per token, [rmsnorm(emb) * enorm | rmsnorm(h) * hnorm] quantized as one 4096-vector.
void k_mtp_in(const StepParams* P, const float* h, const float* enorm, const float* hnorm, float eps, ActQ out, int T,
              cudaStream_t s);

// ---- GEMV (weights read once for all T tokens)
void k_gemv_q8(const DQ8& W, ActQ a, float* y, int T, cudaStream_t s);
// Explicit load selection for parity tests; original Q6 dispatch is unchanged.
void k_test_gemv_q8(const DQ8& W, ActQ a, float* y, int T, bool readonly, cudaStream_t s);
void k_gemv_f32(const DF32& W, const float* x, float* y, int T, cudaStream_t s);

// ---- gated delta net: the T tokens are scanned in order.  With `snap` set, the conv / recurrent state after each
// token but the last is also written to snap + t * snap_stride (floats), for rolling back rejected drafts.
void k_gdn_conv(const float* qkvz, float* conv_state, const float* conv_w, float* conv_out, int T, float* snap,
                int64_t snap_stride, cudaStream_t s);
// Cached strict opt-in setting. The engine may select only prompt calls by passing scratch there.
bool gdn_prepare_enabled();
constexpr size_t gdn_prepare_floats(int tokens) { return (size_t)tokens * 32 * 4; }
// Null scratch retains the original kernel. Nonnull scratch needs gdn_prepare_floats(T)
// floats, 16-byte alignment, no alias with other buffers, and stream-ordered lifetime.
// Explicit pointer selection also permits old/prepared parity in one process.
void k_gdn_recur(const float* conv_out, const float* ba, const float* dt_bias, const float* ssm_a, float* state,
                 float* o, int T, float* snap, int64_t snap_stride, cudaStream_t s, float* prepared_factors = nullptr);
void k_gdn_norm_gate(const float* o, const float* qkvz, const float* norm_w, float eps, ActQ yq, int T, cudaStream_t s);

// ---- gated attention
void k_attn_prep(const float* qkv, const float* q_norm, const float* k_norm, float eps, const StepParams* P,
                 float rope_base, float* q_out, uint16_t* k_cache, uint16_t* v_cache, int max_ctx, int T, cudaStream_t s);
int attn_max_splits(int max_ctx);
void k_attn_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, const StepParams* P, int max_ctx,
                   float* part_o, float* part_ml, int T, cudaStream_t s);
// Optional INT8 KV: caches have [2,max_ctx,256] signed bytes; scales are [2,max_ctx]
// FP32, one per head/token. Causal masks, query layout and split outputs match FP16.
void k_attn_prep_i8(const float* qkv, const float* q_norm, const float* k_norm, float eps, const StepParams* P,
                    float rope_base, float* q_out, int8_t* k_cache, int8_t* v_cache,
                    float* k_scales, float* v_scales, int max_ctx, int T, cudaStream_t s);
void k_attn_decode_i8(const float* q, const int8_t* k_cache, const int8_t* v_cache,
                      const float* k_scales, const float* v_scales, const StepParams* P, int max_ctx,
                      float* part_o, float* part_ml, int T, cudaStream_t s);
void k_attn_combine(const float* part_o, const float* part_ml, const float* qkv, const StepParams* P, int max_ctx,
                    ActQ oq, int T, cudaStream_t s);

// ---- MoE
/// Top-8 routing per token; distinct experts split into resident (HitList) and missing (mailbox for the CPU).
/// ecount[0] / [1] accumulate resident / missing (expert, token) routes, comparable for T=1..4.
void k_router(const float* logits, int layer, const int32_t* residency, const CacheLayerInfo* cinfo, const float* xn,
              HitList* hits, Mailbox* mb_dev, const StepParams* P, int32_t* d_nmiss, float* d_sg, uint32_t* counts,
              unsigned long long* ecount, int T, cudaStream_t s, bool device_mailbox = false);
/// Explicit parity-test selection. omit_system_fence=true requires a device mailbox consumed after kernel completion.
/// Production STRATA_ROUTER_DEVICE_FENCE=1 permits this only when device_mailbox=true; mapped polling stays fenced.
void k_test_router(const float* logits, int layer, const int32_t* residency, const CacheLayerInfo* cinfo, const float* xn,
                   HitList* hits, Mailbox* mb_dev, const StepParams* P, int32_t* d_nmiss, float* d_sg, uint32_t* counts,
                   unsigned long long* ecount, int T, cudaStream_t s, bool omit_system_fence);
void k_swiglu_q(const float* gu, ActQ out, int n_ff, int T, cudaStream_t s);   // out = quant(silu(g) * u), gu = [g | u]
// known_experts=0 keeps the original 8*T launch; a positive host-validated HitList::n removes empty CTAs only.
void k_moe_gu(uint32_t type, ActQ x, const HitList* hits, float* h_gu, int T, cudaStream_t s, int known_experts = 0);
// Combine separately computed resident and transient GPU expert sums before the ordinary CPU/shared combine.
void k_add_expert_output(float* destination, const float* source, int elements, cudaStream_t s);
// Explicit selection for full-kernel parity tests; normal callers use the stable process-level opt-in above.
void k_test_moe_gu(uint32_t type, ActQ x, const HitList* hits, float* h_gu, int T, bool reuse, cudaStream_t s,
                    bool transpose = false, int known_experts = 0);
void k_moe_act(const float* h_gu, const HitList* hits, ActQ hq, int T, cudaStream_t s);
void k_moe_down(uint32_t type, int64_t down_off, ActQ hq, const HitList* hits, float* out, int T, cudaStream_t s);
void k_test_moe_down(uint32_t type, int64_t down_off, ActQ hq, const HitList* hits, float* out, int T, bool reuse,
                      cudaStream_t s, bool transpose = false);
/// x += moe_gpu + moe_cpu (after waiting for it) + sigmoid(sg) * shared; then the next norm.
/// wait_ns accumulates the time the GPU spent waiting for the CPU experts (globaltimer ns).
void k_combine(float* x, const float* moe_gpu, const float* sh_out, const float* d_sg, const int32_t* d_nmiss,
               const Result* res_dev, const StepParams* P, const float* next_w, float eps, float* xn, ActQ xq,
               int32_t* err, unsigned long long* wait_ns, int T, cudaStream_t s);

// ---- head / sampling
/// Argmax of each of T rows of n logits; with prob_dev, also the softmax probability of each argmax.
void k_argmax(const float* logits, int n, int32_t* out_dev, int T, cudaStream_t s, float* prob_dev = nullptr);
void k_candidates(const float* logits, int n, const uint32_t* counts, SamplePen pen, float* tmp, Candidates* out_dev,
                  cudaStream_t s);
/// counts[token[0]] += 1, and counts[token[1]] -= 1 when token[1] >= 0 (the token leaving the penalty window).
void k_count_token(uint32_t* counts, const int32_t* token, cudaStream_t s);

// ---- expert cache maintenance
void k_copy(void* dst, const void* src, size_t bytes, cudaStream_t s);

}  // namespace sq
