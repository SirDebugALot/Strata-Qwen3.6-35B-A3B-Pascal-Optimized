// prefill.cu - batched prompt processing.  See prefill.hpp.
//
// Everything runs on the GPU in chunks of up to C tokens:
//   dense matmuls   : Q8_0 weights dequantized to fp16 per layer, cuBLAS tensor-core GEMM (fp32 accumulate/output)
//   routed experts  : per layer, tokens are grouped by expert on the host; an expert's weights come from its VRAM
//                     slot when resident, otherwise its pinned blob is streamed over PCIe into a staging ring
//                     (copy engine, overlapped with the GEMMs of the previous experts)
//   delta net       : causal conv and the gated delta rule scanned token by token (one or four state columns per warp)
//   attention       : tiled causal attention over the fp16 KV cache
//   MTP             : the MTP layer runs over the chunk's (h_i, token i+1) pairs to fill its KV cache
#include "prefill.hpp"

#include "common.hpp"
#include "engine.hpp"
#include "iq_native.cuh"
#include "quant.hpp"
#if defined(SQ_PREFILL_MMQ)
#include "strata/prefill/moe_mmq.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace sq {

#define CUBLAS_CHECK(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) die("cuBLAS %s failed: %d", #x, (int)st_); } while (0)

namespace {

constexpr int kE = 2048, kHD = 256, kNH = 16, kNKV = 2, kVH = 32, kKH = 16, kDS = 128, kConv = 8192, kFF = 512;
constexpr int kStage = 6;   // staging ring slots for streamed experts
#if defined(SQ_PREFILL_MMQ)
constexpr int kMMQGroup = 16;
constexpr size_t kMMQTail = 4096;

// Explicit compute work avoids routing these tiny clears through a DMA engine. Byte stores make
// no alignment assumption about packed expert strides. One block clears exactly both 4096-byte tails.
__global__ void zero_mmq_tails_kernel(uint8_t* gu_tail, uint8_t* down_tail) {
    for (unsigned i = threadIdx.x; i < kMMQTail; i += blockDim.x) {
        gu_tail[i] = 0;
        down_tail[i] = 0;
    }
}
#endif

// Opt-in layer-major timing. These events are separate from the staging ring's dependency events.
// Stream elapsed time includes scheduling and dependency delays; it is not a kernel-busy counter.
// Allocate once per window, then reuse only after the existing end-of-layer synchronization.
struct LayerMajorProfile {
    cudaEvent_t phase[5] = {};
    std::vector<cudaEvent_t> copy_begin, copy_end, wait_begin, wait_end;
    explicit LayerMajorProfile(bool detail) {
        for (auto& event : phase) CUDA_CHECK(cudaEventCreate(&event));
        if (detail) {
            for (auto* events : {&copy_begin, &copy_end, &wait_begin, &wait_end}) {
                events->resize(256);
                for (auto& event : *events) CUDA_CHECK(cudaEventCreate(&event));
            }
        }
    }
    ~LayerMajorProfile() {
        for (auto event : phase) cudaEventDestroy(event);
        for (const auto* events : {&copy_begin, &copy_end, &wait_begin, &wait_end})
            for (auto event : *events) cudaEventDestroy(event);
    }
    static double elapsed(cudaEvent_t begin, cudaEvent_t end) {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
        return ms;
    }
};

// Independent, opt-in mixer decomposition. Events are allocated once per window and reused only
// after its existing layer synchronization. Unused branch endpoints are never read.
struct MixerProfile {
    enum Point { D2D_BEGIN, D2D_IN, RMS, INPUT, BA, CONV, GDN_PREP, GDN_SCAN, GDN_NORM,
                 ATTN_PREP, ATTN, OUTPUT, D2D_OUT, POINTS };
    enum Metric { DEQUANT_IN_MS, DEQUANT_OUT_MS, D2D_IN_MS, RMS_MS, INPUT_MS, BA_MS, CONV_MS,
                  GDN_PREP_MS, GDN_SCAN_MS, GDN_NORM_MS, ATTN_PREP_MS, ATTN_MS, OUTPUT_MS,
                  D2D_OUT_MS, METRICS };
    cudaEvent_t dequant[3] = {};
    std::vector<cudaEvent_t> events;
    explicit MixerProfile(int tiles) : events((size_t)tiles * POINTS) {
        for (auto& event : dequant) CUDA_CHECK(cudaEventCreate(&event));
        for (auto& event : events) CUDA_CHECK(cudaEventCreate(&event));
    }
    ~MixerProfile() {
        for (auto event : dequant) cudaEventDestroy(event);
        for (auto event : events) cudaEventDestroy(event);
    }
    void mark(int tile, Point point, cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(events[(size_t)tile * POINTS + point], stream));
    }
    static void emit(const char* scope, int layer, int window, int index, int tokens, bool attention,
                     double span, const double* m) {
        log("PREFILL_MIXER_PROFILE scope=%s layer=%d window=%d tile_index=%d tokens=%d attention=%d "
            "stream_span_ms=%.4f dequant_input_ms=%.4f dequant_output_ms=%.4f d2d_in_ms=%.4f rms_ms=%.4f "
            "input_gemm_ms=%.4f ba_gemm_ms=%.4f conv_ms=%.4f gdn_prep_ms=%.4f gdn_scan_ms=%.4f "
            "gdn_normgate_ms=%.4f rope_kv_prep_ms=%.4f attention_ms=%.4f output_gemm_ms=%.4f d2d_out_ms=%.4f",
            scope, layer, window, index, tokens, attention ? 1 : 0, span,
            m[DEQUANT_IN_MS], m[DEQUANT_OUT_MS], m[D2D_IN_MS], m[RMS_MS], m[INPUT_MS], m[BA_MS],
            m[CONV_MS], m[GDN_PREP_MS], m[GDN_SCAN_MS], m[GDN_NORM_MS], m[ATTN_PREP_MS],
            m[ATTN_MS], m[OUTPUT_MS], m[D2D_OUT_MS]);
    }
    void report(int layer, int window, int n, int tile_size, bool attention) const {
        const int tiles = (n + tile_size - 1) / tile_size;
        double total[METRICS] = {};
        total[DEQUANT_IN_MS] = LayerMajorProfile::elapsed(dequant[0], dequant[1]);
        total[DEQUANT_OUT_MS] = LayerMajorProfile::elapsed(dequant[1], dequant[2]);
        for (int t = 0; t < tiles; ++t) {
            const cudaEvent_t* e = events.data() + (size_t)t * POINTS;
            auto dt = [&](Point a, Point b) { return LayerMajorProfile::elapsed(e[a], e[b]); };
            double m[METRICS] = {};
            m[D2D_IN_MS] = dt(D2D_BEGIN, D2D_IN);
            m[RMS_MS] = dt(D2D_IN, RMS);
            m[INPUT_MS] = dt(RMS, INPUT);
            if (attention) {
                m[ATTN_PREP_MS] = dt(INPUT, ATTN_PREP);
                m[ATTN_MS] = dt(ATTN_PREP, ATTN);
                m[OUTPUT_MS] = dt(ATTN, OUTPUT);
            } else {
                m[BA_MS] = dt(INPUT, BA);
                m[CONV_MS] = dt(BA, CONV);
                m[GDN_PREP_MS] = dt(CONV, GDN_PREP);
                m[GDN_SCAN_MS] = dt(GDN_PREP, GDN_SCAN);
                m[GDN_NORM_MS] = dt(GDN_SCAN, GDN_NORM);
                m[OUTPUT_MS] = dt(GDN_NORM, OUTPUT);
            }
            m[D2D_OUT_MS] = dt(OUTPUT, D2D_OUT);
            for (int k = 0; k < METRICS; ++k) total[k] += m[k];
            if (t == 0 || t == tiles - 1)
                emit(t == 0 ? "first" : "last", layer, window, t, std::min(tile_size, n - t * tile_size),
                     attention, dt(D2D_BEGIN, D2D_OUT), m);
        }
        emit("total", layer, window, -1, n, attention,
             LayerMajorProfile::elapsed(dequant[0], events[(size_t)(tiles - 1) * POINTS + D2D_OUT]), total);
    }
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = lane < nw ? sh[lane] : 0.0f;
    return warp_sum(v);
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ __half to_h(float x) { return __float2half_rn(fminf(fmaxf(x, -65504.0f), 65504.0f)); }

// ---------------------------------------------------------------------------------------------- dequantization
__global__ void dequant_q8_kernel(const int8_t* __restrict__ qs, const __half* __restrict__ d, __half* __restrict__ out,
                                  int64_t n /* elements */) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= n) return;
    const float s = __half2float(d[i >> 5]);
    const int2 q = *(const int2*)(qs + i);
    const int8_t* qb = (const int8_t*)&q;
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half_rn(s * (float)qb[k]);
    *(int4*)(out + i) = *(int4*)o;
}

// Keep precisely the existing Q8 -> half rounding, then widen for SGEMM storage. This changes the
// GEMM implementation/reduction order, not the precision of either input operand.
__global__ void dequant_q8_half_f32_kernel(const int8_t* __restrict__ qs, const __half* __restrict__ d,
                                         float* __restrict__ out, int64_t n) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= n) return;
    const float s = __half2float(d[i >> 5]);
    const int2 q = *(const int2*)(qs + i);
    const int8_t* qb = (const int8_t*)&q;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[i + k] = __half2float(__float2half_rn(s * (float)qb[k]));
}

__global__ void widen_half_kernel(const __half* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

__device__ __forceinline__ void sm_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

/// One warp per 256-value super-block; lane l writes values [8l, 8l+8).
template <uint32_t T>
__global__ void dequant_k_kernel(const uint8_t* __restrict__ src, __half* __restrict__ out, int64_t n_sb) {
    const int64_t sb = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (sb >= n_sb) return;
    const int lane = threadIdx.x & 31;
    const int v0 = lane * 8;                 // first value
    __align__(16) __half o[8];
    if constexpr (T == T_Q4_K || T == T_Q5_K) {
        constexpr int BB = T == T_Q4_K ? 144 : 176;
        const uint8_t* b = src + sb * BB;
        const float2 dm = __half22float2(*(const __half2*)b);
        const int j = v0 / 64, hi = (v0 / 32) & 1, l0 = v0 % 32;   // pair, nibble, position in sub-block
        int sc, m;
        sm_k4(2 * j + hi, b + 4, sc, m);
        const float d = dm.x * sc, mn = dm.y * m;
        const uint8_t* qs = b + (T == T_Q4_K ? 16 : 48) + 32 * j + l0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int q = hi ? (qs[k] >> 4) : (qs[k] & 0xF);
            if constexpr (T == T_Q5_K) q += ((b[16 + l0 + k] >> (2 * j + hi)) & 1) << 4;
            o[k] = __float2half_rn(d * q - mn);
        }
    } else {   // Q6_K
        const uint8_t* b = src + sb * 210;
        const float d = __half2float(*(const __half*)(b + 208));
        const int h = v0 / 128, r = v0 % 128, qd = r / 32, l0 = r % 32;
        const uint8_t* ql = b + 64 * h + 32 * (qd & 1) + l0;
        const uint8_t* qh = b + 128 + 32 * h + l0;
        const int8_t sc = ((const int8_t*)(b + 192))[8 * h + 2 * qd + l0 / 16];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lo = (qd >> 1) ? (ql[k] >> 4) : (ql[k] & 0xF);
            const int q = (lo | (((qh[k] >> (2 * qd)) & 3) << 4)) - 32;
            o[k] = __float2half_rn(d * sc * q);
        }
    }
    *(int4*)(out + sb * 256 + v0) = *(int4*)o;
}

void dequant_expert(uint32_t t, const uint8_t* src, __half* out, int64_t n_values, cudaStream_t s) {
    const int64_t nsb = n_values / 256;
    const int blocks = (int)((nsb + 7) / 8);
    switch (t) {
        case T_Q4_K: dequant_k_kernel<T_Q4_K><<<blocks, 256, 0, s>>>(src, out, nsb); break;
        case T_Q5_K: dequant_k_kernel<T_Q5_K><<<blocks, 256, 0, s>>>(src, out, nsb); break;
        case T_Q6_K: dequant_k_kernel<T_Q6_K><<<blocks, 256, 0, s>>>(src, out, nsb); break;
        default: die("dequant_expert: type %u", t);
    }
}

template <uint32_t Type>
__global__ void dequant_native_expert_kernel(const uint8_t* __restrict__ src, __half* __restrict__ out) {
    const int sb = blockIdx.x;
    const int v = threadIdx.x;
    const int group = v >> 5;
    const int within = v & 31;
    float value = 0.0f;
    if constexpr (Type == T_IQ2_S) {
        const block_iq2_s& b = reinterpret_cast<const block_iq2_s*>(src)[sb];
        const int l = within >> 3, j = within & 7;
        const uint8_t* signs = b.qs + 32;
        const int index = b.qs[4 * group + l] | ((b.qh[group] << (8 - 2 * l)) & 0x300);
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(iq2s_grid + index);
        const int nibble = l < 2 ? (b.scales[group] & 15) : (b.scales[group] >> 4);
        const float scale = __half2float(b.d) * (0.5f + nibble) * 0.25f;
        value = scale * grid[j] * ((signs[4 * group + l] >> j) & 1 ? -1.0f : 1.0f);
    } else if constexpr (Type == T_IQ3_S) {
        const block_iq3_s& b = reinterpret_cast<const block_iq3_s*>(src)[sb];
        const int l = within >> 3, j = within & 7;
        const int q = 8 * group + 2 * l + (j >= 4);
        const int shift = j < 4 ? 8 - 2 * l : 7 - 2 * l;
        const int index = b.qs[q] | ((b.qh[group] << shift) & 0x100);
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(iq3s_grid + index);
        const int nibble = (b.scales[group >> 1] >> (4 * (group & 1))) & 15;
        const float scale = __half2float(b.d) * (1 + 2 * nibble);
        value = scale * grid[j & 3] * ((b.signs[4 * group + l] >> j) & 1 ? -1.0f : 1.0f);
    } else if constexpr (Type == T_IQ4_XS) {
        const block_iq4_xs& b = reinterpret_cast<const block_iq4_xs*>(src)[sb];
        const int nibble = ((b.scales_l[group >> 1] >> (4 * (group & 1))) & 15) |
                           (((b.scales_h >> (2 * group)) & 3) << 4);
        const uint8_t q = b.qs[(group << 4) + (within & 15)];
        const int code = within < 16 ? (q & 15) : (q >> 4);
        value = __half2float(b.d) * (nibble - 32) * kvalues_iq4nl[code];
    } else if constexpr (Type == T_Q2_K) {
        const block_q2_K& b = reinterpret_cast<const block_q2_K*>(src)[sb];
        const uint8_t* raw = reinterpret_cast<const uint8_t*>(&b);
        const int g16 = v >> 4, p16 = v & 15, half = g16 >> 3, local = g16 & 7;
        const int q = (b.qs[half * 32 + (local & 1) * 16 + p16] >> (2 * (local >> 1))) & 3;
        const uint8_t sc = b.scales[g16];
        value = __half2float(*reinterpret_cast<const __half*>(raw + 80)) * (sc & 15) * q -
                __half2float(*reinterpret_cast<const __half*>(raw + 82)) * (sc >> 4);
    } else if constexpr (Type == T_Q3_K) {
        const block_q3_K& b = reinterpret_cast<const block_q3_K*>(src)[sb];
        const int g32 = v >> 5, p32 = v & 31;
        const int low = (b.qs[(g32 >> 2) * 32 + p32] >> (2 * (g32 & 3))) & 3;
        const int q = low - ((b.hmask[p32] & (1u << g32)) ? 0 : 4);
        const int scale = iq_native::q3_k_scale(b.scales, 2 * g32 + (within >= 16));
        value = __half2float(b.d) * scale * q;
    }
    out[(size_t)sb * 256 + v] = __float2half_rn(value);
}

void dequant_native_expert(uint32_t t, const uint8_t* src, __half* out, int64_t n_values, cudaStream_t s) {
    const int blocks = (int)(n_values / 256);
    switch (t) {
        case T_IQ2_S:  dequant_native_expert_kernel<T_IQ2_S><<<blocks, 256, 0, s>>>(src, out); break;
        case T_IQ3_S:  dequant_native_expert_kernel<T_IQ3_S><<<blocks, 256, 0, s>>>(src, out); break;
        case T_IQ4_XS: dequant_native_expert_kernel<T_IQ4_XS><<<blocks, 256, 0, s>>>(src, out); break;
        case T_Q2_K:   dequant_native_expert_kernel<T_Q2_K><<<blocks, 256, 0, s>>>(src, out); break;
        case T_Q3_K:   dequant_native_expert_kernel<T_Q3_K><<<blocks, 256, 0, s>>>(src, out); break;
        default: dequant_expert(t, src, out, n_values, s); break;
    }
}

// ---------------------------------------------------------------------------------------------- row kernels
/// Hn = rmsnorm(X) * w (fp32, optional), Hh = same in fp16.  One block of 256 threads per token.
__global__ void rmsnorm_rows_kernel(const float* __restrict__ X, const float* __restrict__ w, float eps,
                                    float* __restrict__ Hn, __half* __restrict__ Hh) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* x = X + (size_t)t * kE;
    float v[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = x[threadIdx.x + 256 * i];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / kE + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = threadIdx.x + 256 * i;
        const float y = v[i] * r * w[c];
        if (Hn) Hn[(size_t)t * kE + c] = y;
        Hh[(size_t)t * kE + c] = to_h(y);
    }
}

/// Hh[t * ldo + c] = fp16(rmsnorm(X[t]) * w)  (one half of the MTP's [enorm(e) | hnorm(h)] input)
__global__ void rmsnorm_rows_ld_kernel(const float* __restrict__ X, const float* __restrict__ w, float eps,
                                       __half* __restrict__ Hh, int ldo) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* x = X + (size_t)t * kE;
    float v[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = x[threadIdx.x + 256 * i];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / kE + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = threadIdx.x + 256 * i;
        Hh[(size_t)t * ldo + c] = to_h(v[i] * r * w[c]);
    }
}

__global__ void gather_rows_kernel(const __half* __restrict__ src, const int* __restrict__ idx, __half* __restrict__ dst) {
    const int r = blockIdx.x;
    const int4* s = (const int4*)(src + (size_t)idx[r] * kE);
    int4* d = (int4*)(dst + (size_t)r * kE);
    d[threadIdx.x] = s[threadIdx.x];   // 256 threads x 8 halves
}

#if defined(SQ_PREFILL_MMQ)
__global__ void mmq_bounds_kernel(int* bounds, int first, int last) {
    bounds[0] = first;
    bounds[1] = last;
}
#endif

/// out[r][i] = silu(gu[r][i]) * gu[r][n + i]  (fp16 out)
__global__ void swiglu_rows_kernel(const float* __restrict__ gu, __half* __restrict__ out, int n) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[(size_t)r * n + i] = to_h(silu(gu[(size_t)r * 2 * n + i]) * gu[(size_t)r * 2 * n + n + i]);
}

/// X[t] += sum_k w[t,k] * moe[inv[t,k]] + sigmoid(sg[t]) * sh[t]
__global__ void moe_combine_rows_kernel(float* __restrict__ X, const float* __restrict__ moe, const int* __restrict__ inv,
                                        const float* __restrict__ w, const float* __restrict__ sh,
                                        const float* __restrict__ rlog, int ldr) {
    const int t = blockIdx.x;
    const float sg = sigm(rlog[(size_t)t * ldr + 256]);
    int a[8];
    float ww[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { a[k] = inv[t * 8 + k]; ww[k] = w[t * 8 + k]; }
    for (int c = threadIdx.x; c < kE; c += blockDim.x) {
        float acc = sg * sh[(size_t)t * kE + c];
#pragma unroll
        for (int k = 0; k < 8; ++k) acc += ww[k] * moe[(size_t)a[k] * kE + c];
        X[(size_t)t * kE + c] += acc;
    }
}

/// Layer-major phase 1: shared expert is complete while routed experts are intentionally deferred.
__global__ void shared_combine_rows_kernel(float* __restrict__ X, const float* __restrict__ sh,
                                           const float* __restrict__ rlog, int ldr) {
    const int t = blockIdx.x;
    const float sg = sigm(rlog[(size_t)t * ldr + 256]);
    for (int c = threadIdx.x; c < kE; c += blockDim.x)
        X[(size_t)t * kE + c] += sg * sh[(size_t)t * kE + c];
}

__global__ void shared_store_rows_kernel(float* __restrict__ out, const float* __restrict__ sh,
                                         const float* __restrict__ rlog, int ldr) {
    const int t = blockIdx.x;
    const float sg = sigm(rlog[(size_t)t * ldr + 256]);
    for (int c = threadIdx.x; c < kE; c += blockDim.x)
        out[(size_t)t * kE + c] = sg * sh[(size_t)t * kE + c];
}

/// Layer-major phase 2: add the routed rows held by one bounded expert group to the full-prompt residual.
__global__ void routed_group_add_kernel(float* __restrict__ X, const float* __restrict__ moe,
                                        const int* __restrict__ inv, const float* __restrict__ w,
                                        int T, int first, int last) {
    const int t = blockIdx.x;
    if (t >= T) return;
    int pos[8];
    float ww[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { pos[k] = inv[t * 8 + k]; ww[k] = w[t * 8 + k]; }
    for (int c = threadIdx.x; c < kE; c += blockDim.x) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            if (pos[k] >= first && pos[k] < last) acc += ww[k] * moe[(size_t)(pos[k] - first) * kE + c];
        X[(size_t)t * kE + c] += acc;
    }
}

__global__ void add_rows_kernel(float* __restrict__ X, const float* __restrict__ add, int T) {
    const int t = blockIdx.x;
    if (t >= T) return;
    for (int c = threadIdx.x; c < kE; c += blockDim.x)
        X[(size_t)t * kE + c] += add[(size_t)t * kE + c];
}

/// Router for a batch: softmax over 256 experts, top-8, weights renormalized.  One block (256) per token.
__global__ void router_rows_kernel(const float* __restrict__ rlog, int ldr, int* __restrict__ ids, float* __restrict__ wts,
                                   uint32_t* __restrict__ counts, int layer) {
    __shared__ float sh[32];
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ int sel[8];
    __shared__ float selp[8];
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float l = rlog[(size_t)t * ldr + e];
    float m = warp_max(l);
    if (lane == 0) bv[warp] = m;
    __syncthreads();
    m = bv[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, bv[i]);
    const float ex = __expf(l - m);
    const float tot = block_sum(ex, sh);
    float p = ex / tot;
    for (int k = 0; k < 8; ++k) {
        float v = p;
        int idx = e;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        __syncthreads();
        if (lane == 0) { bv[warp] = v; bi[warp] = idx; }
        __syncthreads();
        if (e == 0) {
            float bvv = bv[0];
            int bii = bi[0];
            for (int i = 1; i < 8; ++i)
                if (bv[i] > bvv || (bv[i] == bvv && bi[i] < bii)) { bvv = bv[i]; bii = bi[i]; }
            sel[k] = bii;
            selp[k] = bvv;
        }
        __syncthreads();
        if (e == sel[k]) p = -1.0f;
    }
    if (e < 8) {
        float s = 0.0f;
        for (int k = 0; k < 8; ++k) s += selp[k];
        ids[t * 8 + e] = sel[e];
        wts[t * 8 + e] = selp[e] / s;
        atomicAdd(&counts[layer * 256 + sel[e]], 1u);
    }
}

// ---------------------------------------------------------------------------------------------- delta net
/// Causal depthwise conv over the chunk (kernel 4) + SiLU; the conv state holds the previous 3 inputs.
__global__ void conv_seq_kernel(const float* __restrict__ qkvz, int ld, float* __restrict__ cs, const float* __restrict__ cw,
                                float* __restrict__ out, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kConv) return;
    const float4 w = *(const float4*)(cw + c * 4);
    float s0 = cs[c * 3 + 0], s1 = cs[c * 3 + 1], s2 = cs[c * 3 + 2];
    for (int t = 0; t < T; ++t) {
        const float in = qkvz[(size_t)t * ld + c];
        const float y = w.x * s0 + w.y * s1 + w.z * s2 + w.w * in;
        s0 = s1;
        s1 = s2;
        s2 = in;
        out[(size_t)t * kConv + c] = silu(y);
    }
    cs[c * 3 + 0] = s0;
    cs[c * 3 + 1] = s1;
    cs[c * 3 + 2] = s2;
}

/// Per token: l2-normalize q and k heads in place (q also takes the 1/sqrt(128) scale); decay and beta per v head.
__global__ void gdn_prep_kernel(float* __restrict__ conv, const float* __restrict__ ba, const float* __restrict__ dt_bias,
                                const float* __restrict__ ssm_a, float* __restrict__ gb) {
    const int t = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;   // 32 warps: 16 q heads + 16 k heads
    float* p = conv + (size_t)t * kConv + warp * kDS;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] = p[lane + 32 * i]; ss += v[i] * v[i]; }
    ss = warp_sum(ss);
    const float r = rsqrtf(ss + 1e-6f) * (warp < kKH ? rsqrtf((float)kDS) : 1.0f);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[lane + 32 * i] = v[i] * r;
    if (threadIdx.x < kVH) {
        const int h = threadIdx.x;
        const float a = ba[t * 64 + kVH + h] + dt_bias[h];
        const float sp = a > 20.0f ? a : log1pf(__expf(a));
        gb[t * 64 + h] = __expf(sp * ssm_a[h]);         // decay
        gb[t * 64 + kVH + h] = sigm(ba[t * 64 + h]);   // beta
    }
}

/// grid (32 v heads, 32 column groups), 4 warps; each warp owns one state column and scans the T tokens.
__global__ void __launch_bounds__(128) gdn_scan_kernel(const float* __restrict__ conv, const float* __restrict__ gb,
                                                       float* __restrict__ state, float* __restrict__ o, int T) {
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = blockIdx.y * 4 + warp;
    const int kh = h % kKH;
    float* S = state + ((size_t)h * kDS + col) * kDS;
    float s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = S[lane + 32 * i];
    for (int t = 0; t < T; ++t) {
        const float* ct = conv + (size_t)t * kConv;
        float q[4], k[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = ct[kh * kDS + lane + 32 * i];
            k[i] = ct[kKH * kDS + kh * kDS + lane + 32 * i];
        }
        const float v = ct[2 * kKH * kDS + h * kDS + col];
        const float decay = gb[t * 64 + h], beta = gb[t * 64 + kVH + h];
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) kv += s[i] * k[i];
        kv = warp_sum(kv);
        const float delta = (v - decay * kv) * beta;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] = decay * s[i] + k[i] * delta;
            out += s[i] * q[i];
        }
        out = warp_sum(out);
        if (lane == 0) o[(size_t)t * (kVH * kDS) + h * kDS + col] = out;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) S[lane + 32 * i] = s[i];
}

// Reuse each token's q/k/decay/beta across four independent state columns. Keep the original
// per-column accumulation, warp reduction and token order: no approximation or extra workspace.
// Four columns won the Pascal probe over two/eight; the original kernel remains the default.
__global__ void __launch_bounds__(128) gdn_scan_four_columns_kernel(
        const float* __restrict__ conv, const float* __restrict__ gb,
        float* __restrict__ state, float* __restrict__ o, int T) {
    constexpr int C = 4;
    static_assert(kDS % (4 * C) == 0, "GDN grid must cover complete state columns");
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int first = (blockIdx.y * 4 + warp) * C, kh = h % kKH;
    float s[C][4];
#pragma unroll
    for (int c = 0; c < C; ++c) {
        const float* S = state + ((size_t)h * kDS + first + c) * kDS;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[c][i] = S[lane + 32 * i];
    }
    for (int t = 0; t < T; ++t) {
        const float* ct = conv + (size_t)t * kConv;
        float q[4], k[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = ct[kh * kDS + lane + 32 * i];
            k[i] = ct[kKH * kDS + kh * kDS + lane + 32 * i];
        }
        const float decay = gb[t * 64 + h], beta = gb[t * 64 + kVH + h];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const int col = first + c;
            const float v = ct[2 * kKH * kDS + h * kDS + col];
            float kv = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) kv += s[c][i] * k[i];
            kv = warp_sum(kv);
            const float delta = (v - decay * kv) * beta;
            float out = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[c][i] = decay * s[c][i] + k[i] * delta;
                out += s[c][i] * q[i];
            }
            out = warp_sum(out);
            if (lane == 0) o[(size_t)t * (kVH * kDS) + h * kDS + col] = out;
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float* S = state + ((size_t)h * kDS + first + c) * kDS;
#pragma unroll
        for (int i = 0; i < 4; ++i) S[lane + 32 * i] = s[c][i];
    }
}

void gdn_scan(const float* conv, const float* gb, float* state, float* out, int T,
              int columns, cudaStream_t stream) {
    if (columns == 4)
        gdn_scan_four_columns_kernel<<<dim3(kVH, kDS / 16), 128, 0, stream>>>(conv, gb, state, out, T);
    else
        gdn_scan_kernel<<<dim3(kVH, kDS / 4), 128, 0, stream>>>(conv, gb, state, out, T);
}

/// Y[t][h] = rmsnorm(o[t][h]) * w * silu(z[t][h]) in fp16.  grid (T, 32), 128 threads.
__global__ void gdn_normgate_rows_kernel(const float* __restrict__ o, const float* __restrict__ qkvz, int ld,
                                         const float* __restrict__ nw, float eps, __half* __restrict__ Y) {
    __shared__ float sh[32];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const float v = o[(size_t)t * 4096 + h * kDS + i];
    const float ss = block_sum(v * v, sh);
    const float r = rsqrtf(ss / kDS + eps);
    const float z = qkvz[(size_t)t * ld + kConv + h * kDS + i];
    Y[(size_t)t * 4096 + h * kDS + i] = to_h(v * r * nw[i] * silu(z));
}

// ---------------------------------------------------------------------------------------------- attention
/// Per token: q/k rmsnorm + rope; q (pre-scaled) -> Qh fp16 [T][16][256]; k, v -> caches at p0 + t.
/// grid (T, 20): 16 q heads, 2 k heads, 2 v heads.
__global__ void attn_prep_rows_kernel(const float* __restrict__ qkv, int ld, const float* __restrict__ qnw,
                                      const float* __restrict__ knw, float eps, int p0, float base,
                                      __half* __restrict__ Qh, __half* __restrict__ kc, __half* __restrict__ vc, int max_ctx) {
    __shared__ float sh[32];
    __shared__ float sv[kHD];
    const int t = blockIdx.x, b = blockIdx.y, i = threadIdx.x;
    const float* row = qkv + (size_t)t * ld;
    const int pos = p0 + t;
    if (b >= kNH + kNKV) {
        const int kh = b - kNH - kNKV;
        vc[((size_t)kh * max_ctx + pos) * kHD + i] = __float2half_rn(row[kNH * 2 * kHD + kNKV * kHD + kh * kHD + i]);
        return;
    }
    const bool isq = b < kNH;
    const float x = isq ? row[b * 2 * kHD + i] : row[kNH * 2 * kHD + (b - kNH) * kHD + i];
    const float ss = block_sum(x * x, sh);
    const float r = rsqrtf(ss / kHD + eps);
    sv[i] = x * r * (isq ? qnw[i] : knw[i]);
    __syncthreads();
    if (i < 32) {
        const double inv = pow((double)base, -2.0 * i / 64.0);
        double sn, cs;
        sincos((double)pos * (double)(float)inv, &sn, &cs);
        const float x0 = sv[i], x1 = sv[i + 32];
        sv[i] = x0 * (float)cs - x1 * (float)sn;
        sv[i + 32] = x0 * (float)sn + x1 * (float)cs;
    }
    __syncthreads();
    if (isq) Qh[((size_t)t * kNH + b) * kHD + i] = __float2half_rn(sv[i] * 0.0625f);
    else kc[((size_t)(b - kNH) * max_ctx + pos) * kHD + i] = __float2half_rn(sv[i]);
}

constexpr int kAK = 32;               // keys per tile
constexpr int kKPad = kHD + 8;        // padded K row (halves) against bank conflicts

// AQ=8 uses 75,008 bytes of shared memory. Pascal has a 48 KiB per-block limit, so AQ=2 uses 43,712
// bytes and 64 threads with the same score and softmax operations for each query.
template <int AQ, bool SHARE_KV = false>
constexpr size_t attn_smem() {
    constexpr int AR = AQ * 8;
    return (size_t)AR * kHD * 2 + (size_t)kAK * kKPad * 2 + (SHARE_KV ? 0 : (size_t)kAK * kHD * 2) +
           (size_t)AR * kAK * 4 + 3 * AR * 4;
}

/// Causal attention for a chunk. grid (ceil(T/AQ), 2 kv heads), AQ*32 threads. Rows r = token*8 + head.
template <int AQ, bool SHARE_KV = false>
__global__ void __launch_bounds__(256) attn_prefill_kernel(const __half* __restrict__ Qh, const __half* __restrict__ kc,
                                                           const __half* __restrict__ vc, int T, int p0, int max_ctx,
                                                           const float* __restrict__ qkv, int ld, __half* __restrict__ out) {
    constexpr int kAQ = AQ, kAR = AQ * 8;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* sQ = (__half*)smem_raw;                          // [64][256]
    __half* sK = sQ + kAR * kHD;                              // [32][264]
    __half* sV = SHARE_KV ? sK : sK + kAK * kKPad;             // [32][256], optionally aliases K
    float* sS = (float*)(SHARE_KV ? sK + kAK * kKPad : sV + kAK * kHD); // after max(K,V) when shared
    float* sM = sS + kAR * kAK;                               // [64] running max
    float* sL = sM + kAR;                                     // [64] running sum
    float* sA = sL + kAR;                                     // [64] rescale factor
    const int kh = blockIdx.y, tq0 = blockIdx.x * kAQ, tid = threadIdx.x;
    const int nq = min(kAQ, T - tq0);
    // load Q rows: r = tok*8 + h  ->  Qh[tq0+tok][kh*8+h]
    for (int i = tid; i < kAR * kHD / 8; i += AQ * 32) {
        const int r = i / (kHD / 8), c8 = i % (kHD / 8);
        const int tok = r >> 3, h = r & 7;
        int4 v = make_int4(0, 0, 0, 0);
        if (tok < nq) v = *(const int4*)(Qh + ((size_t)(tq0 + tok) * kNH + kh * 8 + h) * kHD + c8 * 8);
        *(int4*)(sQ + r * kHD + c8 * 8) = v;
    }
    if (tid < kAR) { sM[tid] = -INFINITY; sL[tid] = 0.0f; }
    // output accumulators: thread -> rows rg*8 .. rg*8+7 (rg = warp), dims lane*8 .. lane*8+7
    const int warp = tid >> 5, lane = tid & 31;
    float acc[8][8];
#pragma unroll
    for (int a = 0; a < 8; ++a)
#pragma unroll
        for (int b = 0; b < 8; ++b) acc[a][b] = 0.0f;
    const int last_pos = p0 + tq0 + nq - 1;
    const int n_keys = last_pos + 1;
    // score micro-tile: rows (rp*2, rp*2+1), keys kq*4 .. kq*4+3
    const int rp = tid >> 3, kq = tid & 7;
    for (int k0 = 0; k0 < n_keys; k0 += kAK) {
        __syncthreads();
        for (int i = tid; i < kAK * kHD / 8; i += AQ * 32) {
            const int j = i / (kHD / 8), c8 = i % (kHD / 8);
            int4 kv4 = make_int4(0, 0, 0, 0), vv4 = make_int4(0, 0, 0, 0);
            if (k0 + j < n_keys) {
                kv4 = *(const int4*)(kc + ((size_t)kh * max_ctx + k0 + j) * kHD + c8 * 8);
                if constexpr (!SHARE_KV)
                    vv4 = *(const int4*)(vc + ((size_t)kh * max_ctx + k0 + j) * kHD + c8 * 8);
            }
            *(int4*)(sK + j * kKPad + c8 * 8) = kv4;
            if constexpr (!SHARE_KV) *(int4*)(sV + j * kHD + c8 * 8) = vv4;
        }
        __syncthreads();
        {
            float s[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            const __half2* q0 = (const __half2*)(sQ + (rp * 2) * kHD);
            const __half2* q1 = (const __half2*)(sQ + (rp * 2 + 1) * kHD);
#pragma unroll 8
            for (int d2 = 0; d2 < kHD / 2; ++d2) {
                const float2 a0 = __half22float2(q0[d2]), a1 = __half22float2(q1[d2]);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float2 kf = __half22float2(((const __half2*)(sK + (kq * 4 + u) * kKPad))[d2]);
                    s[0][u] += a0.x * kf.x + a0.y * kf.y;
                    s[1][u] += a1.x * kf.x + a1.y * kf.y;
                }
            }
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const int r = rp * 2 + a;
                const int qpos = p0 + tq0 + (r >> 3);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int kp = k0 + kq * 4 + u;
                    sS[r * kAK + kq * 4 + u] = (kp <= qpos && (r >> 3) < nq) ? s[a][u] : -INFINITY;
                }
            }
        }
        __syncthreads();
        {   // online softmax: 4 threads per row, 8 scores each
            const int r = tid >> 2, part = tid & 3;
            float* row = sS + r * kAK + part * 8;
            float mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < 8; ++u) mx = fmaxf(mx, row[u]);
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float m_old = sM[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const float e = m_new == -INFINITY ? 0.0f : __expf(row[u] - m_new);
                row[u] = e;
                sum += e;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            __syncwarp();
            if (part == 0) {
                const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                sA[r] = alpha;
                sL[r] = sL[r] * alpha + sum;
                sM[r] = m_new;
            }
        }
        __syncthreads();
        if constexpr (SHARE_KV) {
            // The preceding barrier covers every K reader and softmax writer. K is now dead;
            // overwrite its larger padded allocation with V, then publish V to every accumulator.
            for (int i = tid; i < kAK * kHD / 8; i += AQ * 32) {
                const int j = i / (kHD / 8), c8 = i % (kHD / 8);
                int4 vv4 = make_int4(0, 0, 0, 0);
                if (k0 + j < n_keys)
                    vv4 = *(const int4*)(vc + ((size_t)kh * max_ctx + k0 + j) * kHD + c8 * 8);
                *(int4*)(sV + j * kHD + c8 * 8) = vv4;
            }
            __syncthreads();
        }
#pragma unroll
        for (int a = 0; a < 8; ++a) {
            const float al = sA[warp * 8 + a];
#pragma unroll
            for (int b = 0; b < 8; ++b) acc[a][b] *= al;
        }
        const int kn = min(kAK, n_keys - k0);
        for (int j = 0; j < kn; ++j) {
            const int4 raw = *(const int4*)(sV + j * kHD + lane * 8);
            const __half2* hv = (const __half2*)&raw;
            float vf[8];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 f = __half22float2(hv[u]);
                vf[2 * u] = f.x;
                vf[2 * u + 1] = f.y;
            }
#pragma unroll
            for (int a = 0; a < 8; ++a) {
                const float p = sS[(warp * 8 + a) * kAK + j];
#pragma unroll
                for (int b = 0; b < 8; ++b) acc[a][b] += p * vf[b];
            }
        }
    }
    __syncthreads();
    // write: out[t][h*256 + d] = acc / l * sigmoid(gate)
#pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int r = warp * 8 + a;
        const int tok = r >> 3, h = kh * 8 + (r & 7);
        if (tok >= nq) continue;
        const float inv_l = 1.0f / sL[r];
        const float* gate = qkv + (size_t)(tq0 + tok) * ld + h * 2 * kHD + kHD;
        __align__(16) __half o[8];
#pragma unroll
        for (int b = 0; b < 8; ++b) o[b] = to_h(acc[a][b] * inv_l * sigm(gate[lane * 8 + b]));
        *(int4*)(out + (size_t)(tq0 + tok) * (kNH * kHD) + h * kHD + lane * 8) = *(int4*)o;
    }
}

#include "attn_prefill_banks.cuh"

template <int AQ, bool SHARE_KV = false>
void launch_attn_variant(const __half* q, const __half* k, const __half* v, int T, int p0, int max_ctx,
                         const float* qkv, int ld, __half* out, cudaStream_t stream) {
    attn_prefill_kernel<AQ, SHARE_KV><<<dim3((T + AQ - 1) / AQ, kNKV), AQ * 32,
                                      attn_smem<AQ, SHARE_KV>(), stream>>>(q, k, v, T, p0, max_ctx, qkv, ld, out);
}

// Both the prepared main-layer path and ordinary/MTP path use this dispatcher.
void launch_attn_prefill(bool shared_kv, int aq, const __half* q, const __half* k, const __half* v,
                        int T, int p0, int max_ctx, const float* qkv, int ld, __half* out, cudaStream_t stream) {
    if (!shared_kv) {
        if (aq == 2) launch_attn_variant<2>(q, k, v, T, p0, max_ctx, qkv, ld, out, stream);
        else launch_attn_variant<8>(q, k, v, T, p0, max_ctx, qkv, ld, out, stream);
        return;
    }
    switch (aq) {
        case 2: launch_attn_variant<2, true>(q, k, v, T, p0, max_ctx, qkv, ld, out, stream); break;
        case 3: launch_attn_variant<3, true>(q, k, v, T, p0, max_ctx, qkv, ld, out, stream); break;
        case 4: launch_attn_variant<4, true>(q, k, v, T, p0, max_ctx, qkv, ld, out, stream); break;
        case 6: launch_attn_variant<6, true>(q, k, v, T, p0, max_ctx, qkv, ld, out, stream); break;
        default: SQ_CHECK(false, "invalid shared-KV attention query tile %d", aq);
    }
}

template <int AQ>
void configure_shared_attn(const cudaDeviceProp& prop, size_t shared_limit) {
    constexpr size_t bytes = attn_smem<AQ, true>();
    SQ_CHECK(AQ * 32 <= prop.maxThreadsPerBlock && AQ * 32 <= prop.maxThreadsDim[0],
             "shared-KV attention AQ=%d needs %d threads per block", AQ, AQ * 32);
    SQ_CHECK(bytes <= shared_limit, "shared-KV attention AQ=%d needs %zu bytes, device supports %zu", AQ, bytes, shared_limit);
    if (bytes > prop.sharedMemPerBlock)
        CUDA_CHECK(cudaFuncSetAttribute(attn_prefill_kernel<AQ, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    log("prompt attention: shared K/V, query tile %d, %d threads, %zu bytes shared memory for %s", AQ, AQ * 32, bytes, prop.name);
}

}  // namespace

// =================================================================================================================
struct Prefill::Impl {
    int C = 0;
    bool small_attn = false;
    bool shared_attn_kv = false;
    bool attn_bank_layout = false;
    int attn_aq = 0;
    bool layer_major = false;
    int layer_tile = 1024;
    bool compact_workspace = false;
    bool dense_f32 = false;
    bool dense_algo6 = false; // Opt-in, prepared main input projections on sm61 only.
    int gdn_columns = 1;
    std::string diagnostic_dir;
    int diagnostic_layer = -1;
    cublasHandle_t cb = nullptr;
    cudaStream_t s = nullptr, cs = nullptr;
    MixerProfile* mixer_profile = nullptr; // Non-owning; active only while a sampled layer is enqueued.
    int mixer_profile_tile = 0;
    // activations
    float *X = nullptr, *Hn = nullptr, *big = nullptr, *ba = nullptr, *conv = nullptr, *gb = nullptr, *O = nullptr;
    float *rlog = nullptr, *moe = nullptr, *gu = nullptr, *shgu = nullptr, *shout = nullptr, *wts = nullptr;
    __half *Hh = nullptr, *Y = nullptr, *W16 = nullptr, *W16b = nullptr, *Xe = nullptr, *act = nullptr, *Wgu = nullptr, *Wdn = nullptr, *Qh = nullptr;
    int *ids = nullptr, *perm = nullptr, *inv = nullptr;
    // host (pinned)
    int* h_ids = nullptr;
    int* h_perm = nullptr;   // [C*8] perm then [C*8] inv
    // staging ring for streamed experts
    uint8_t* stage = nullptr;
    int64_t stage_bytes = 0;
    bool stage_alias = false;
    cudaEvent_t ev_copied[kStage], ev_free[kStage];
#if defined(SQ_PREFILL_MMQ)
    bool use_mmq = false;
    bool group_prefetch = false;
    std::vector<char> mmq_layer;
    std::unique_ptr<strata::prefill::mmq::Context> mmq;
    uint8_t *Xq = nullptr, *Hq = nullptr, *mmq_gu = nullptr, *mmq_down = nullptr;
    uint8_t *mmq_gu_alt = nullptr, *mmq_down_alt = nullptr;
    cudaEvent_t mmq_ready[2] = {}, mmq_free[2] = {};
    int *mmq_identity = nullptr, *mmq_bounds = nullptr;
    int *h_mmq_bounds = nullptr;
    size_t mmq_bounds_capacity = 1024;
    size_t xq_stride = 0, hq_stride = 0;
    size_t mmq_gu_stride = 0, mmq_down_stride = 0;
#endif
};

Prefill::~Prefill() {
    if (p_) {
        if (p_->s) cudaStreamSynchronize(p_->s);
        if (p_->cs) cudaStreamSynchronize(p_->cs);
        for (int i = 0; i < kStage; ++i) {
            if (p_->ev_copied[i]) cudaEventDestroy(p_->ev_copied[i]);
            if (p_->ev_free[i]) cudaEventDestroy(p_->ev_free[i]);
        }
        if (p_->cb) cublasDestroy(p_->cb);
        for (void* q : {(void*)p_->X, (void*)p_->Hn, (void*)p_->big, (void*)p_->ba, (void*)p_->conv,
                        (void*)p_->gb, (void*)p_->rlog, (void*)(p_->compact_workspace ? nullptr : p_->gu),
                        (void*)p_->shgu, (void*)p_->shout,
                        (void*)p_->wts, (void*)p_->Hh, (void*)p_->Y, (void*)p_->W16, (void*)p_->W16b, (void*)p_->Xe,
                        (void*)p_->act, (void*)p_->Wgu, (void*)p_->Wdn,
                        (void*)(p_->compact_workspace ? nullptr : p_->Qh), (void*)p_->ids,
                        (void*)p_->perm, (void*)(p_->stage_alias ? nullptr : p_->stage)})
            if (q) cudaFree(q);
#if defined(SQ_PREFILL_MMQ)
        p_->mmq.reset();
        for (int i = 0; i < 2; ++i) {
            if (p_->mmq_ready[i]) cudaEventDestroy(p_->mmq_ready[i]);
            if (p_->mmq_free[i]) cudaEventDestroy(p_->mmq_free[i]);
        }
        for (void* q : {(void*)p_->Xq, (void*)p_->Hq, (void*)p_->mmq_gu, (void*)p_->mmq_down,
                        (void*)p_->mmq_gu_alt, (void*)p_->mmq_down_alt,
                        (void*)p_->mmq_identity, (void*)p_->mmq_bounds})
            if (q) cudaFree(q);
        if (p_->h_mmq_bounds) cudaFreeHost(p_->h_mmq_bounds);
#endif
        if (p_->h_ids) cudaFreeHost(p_->h_ids);
        if (p_->h_perm) cudaFreeHost(p_->h_perm);
        delete p_;
    }
}

bool Prefill::init(std::string& err) {
    (void)err;
    p_ = new Impl;
    Impl& P = *p_;
    const Config& c = e_.model_.cfg;
    if (const char* v = std::getenv("STRATA_PREFILL_LAYER_MAJOR")) P.layer_major = std::strcmp(v, "0") != 0;
    if (const char* v = std::getenv("STRATA_PREFILL_DENSE_F32"))
        P.dense_f32 = P.layer_major && *v && std::strcmp(v, "0") != 0;
    if (const char* v = std::getenv("STRATA_PREFILL_LAYER_TILE")) P.layer_tile = std::atoi(v);
    if (const char* v = std::getenv("STRATA_PREFILL_GDN_COLUMNS")) P.gdn_columns = std::atoi(v);
    SQ_CHECK(P.gdn_columns == 1 || P.gdn_columns == 4, "STRATA_PREFILL_GDN_COLUMNS must be 1 or 4");
    if (P.gdn_columns == 4) log("prefill GDN: 4 state columns per warp, shared q/k loads, no additional workspace");
    if (const char* v = std::getenv("STRATA_PREFILL_DIAGNOSTIC_DIR")) P.diagnostic_dir = v;
    if (const char* v = std::getenv("STRATA_PREFILL_DIAGNOSTIC_LAYER")) P.diagnostic_layer = std::atoi(v);
    if (!P.diagnostic_dir.empty())
        log("prefill hidden diagnostics: %s, first 1024 rows at position 0, main layer %d (-1=all)",
            P.diagnostic_dir.c_str(), P.diagnostic_layer);
    SQ_CHECK(!P.layer_major || (P.layer_tile >= 128 && P.layer_tile <= 2048 && P.layer_tile % 128 == 0),
             "STRATA_PREFILL_LAYER_TILE must be a multiple of 128 from 128 to 2048");
    // with MTP, a chunk's MTP pass also carries up to kMaxT pending pairs of the previous step: room for them
    P.C = (P.layer_major ? P.layer_tile : std::max(16, e_.opt_.prefill_chunk)) + (e_.mtp_on_ ? kMaxT : 0);
    const int C = P.C;
    P.s = e_.stream_;
    P.cs = e_.copy_stream_;
#if defined(SQ_PREFILL_MMQ)
    if (const char* v = std::getenv("STRATA_PREFILL_GROUP_PREFETCH"))
        P.group_prefetch = P.layer_major && *v && std::strcmp(v, "0") != 0;
    P.mmq_layer.assign(e_.model_.layers.size(), 0);
    if (const char* v = std::getenv("STRATA_PREFILL_MMQ"); v && *v && std::strcmp(v, "0") != 0) {
        for (size_t i = 0; i < e_.model_.layers.size(); ++i) {
            const LayerW& L = e_.model_.layers[i];
            P.mmq_layer[i] = strata::prefill::mmq::fits((int)L.t_gu, 2 * kFF) &&
                             strata::prefill::mmq::fits((int)L.t_down, kE);
            P.use_mmq |= P.mmq_layer[i] != 0;
        }
    }
    if (const char* v = std::getenv("STRATA_PREFILL_WORKSPACE_COMPACT"); v && *v && std::strcmp(v, "0") != 0) {
        P.compact_workspace = P.layer_major && P.use_mmq &&
            std::all_of(P.mmq_layer.begin(), P.mmq_layer.end(), [](char x) { return x != 0; });
        if (!P.compact_workspace)
            log("compact prefill workspace not enabled: requires layer-major MMQ coverage for all main and MTP layers");
    }
#endif
    CUBLAS_CHECK(cublasCreate(&P.cb));
    CUBLAS_CHECK(cublasSetStream(P.cb, P.s));
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUBLAS_CHECK(cublasSetMathMode(P.cb, prop.major >= 7 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
    if (const char* v = std::getenv("STRATA_PREFILL_DENSE_ALGO6"); v && std::strcmp(v, "1") == 0) {
#if !defined(__HIPCC__) && !defined(GGML_USE_HIP)
        P.dense_algo6 = P.layer_major && P.dense_f32 && prop.major == 6 && prop.minor == 1;
        if (P.dense_algo6) {
            log("experimental prompt input GEMM: FP32 algorithm 6 on sm61, main layers only, "
                "T=1024/K=2048/M=12288 or 9216; same buffers, no additional workspace; "
                "other shapes/tails use SGEMM because only these full-tile shapes passed bitwise screening");
        } else {
            log("prompt FP32 algorithm 6 not enabled: requires layer-major dense FP32 input projections on sm61");
        }
#else
        log("prompt FP32 algorithm 6 not enabled: CUDA sm61 experiment is unavailable in HIP builds");
#endif
    }
    size_t free_before = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));
    auto fal = [](size_t n) { float* p; CUDA_CHECK(cudaMalloc(&p, n * 4)); return p; };
    auto hal = [](size_t n) { __half* p; CUDA_CHECK(cudaMalloc(&p, n * 2)); return p; };
    auto ial = [](size_t n) { int* p; CUDA_CHECK(cudaMalloc(&p, n * 4)); return p; };
    const size_t big_cols = std::max<size_t>(c.conv_dim() + c.ssm_d_inner, (size_t)c.n_head * c.head_dim * 2 + 2 * c.n_head_kv * c.head_dim);
    P.X = fal((size_t)C * kE);
    P.Hn = fal((size_t)C * kE);
    // big (projections) and O (mixer output) are dead during the MoE half of a layer and moe (per-slot expert
    // outputs) is dead outside it: one allocation serves both, which keeps large chunks affordable.
    P.big = fal((size_t)C * std::max<size_t>(big_cols + 4096, (size_t)8 * kE));
    P.O = P.big + (size_t)C * big_cols;
    P.moe = P.big;
    P.ba = fal((size_t)C * 64);
    P.conv = fal((size_t)C * kConv);
    P.gb = fal((size_t)C * 64);
    P.rlog = fal((size_t)C * 257);
    if (P.compact_workspace) {
        // Conv is live only in a GDN mixer, Qh only in an attention mixer, and gu only in the following MoE.
        // All consumers use P.s, including ordinary MTP's layer(), so their uses cannot overlap. Conv owns
        // the allocation; neither aliased view is freed separately.
        SQ_CHECK((size_t)2 * kFF * 8 <= kConv && (size_t)kNH * kHD * sizeof(__half) <= (size_t)kConv * sizeof(float),
                 "compact mixer/MoE aliases exceed the convolution allocation");
        P.gu = P.conv;
        P.Qh = reinterpret_cast<__half*>(P.conv);
    } else {
        P.gu = fal((size_t)C * 2 * kFF
#if defined(SQ_PREFILL_MMQ)
                   * (P.use_mmq ? 8 : 1)
#endif
        );
    }
    P.shgu = fal((size_t)C * 2 * kFF);
    P.shout = fal((size_t)C * kE);
    P.wts = fal((size_t)C * 8);
    P.Hh = hal((size_t)C * kE);
    P.Y = hal((size_t)C * 4096);
    // Prepared main-layer input projections can use W16 as FP32. Shared experts and ordinary MTP
    // reuse it as FP16 later. Output projections stay FP16: widening did not improve their timing.
    const size_t dense_storage = P.dense_f32 ? 2 : 1;
    P.W16 = hal(big_cols * kE * dense_storage);
    if (P.layer_major) P.W16b = hal((size_t)4096 * kE);
    // Every covered MMQ layer uses Xe only as FP32 swiglu rows [C*8][kFF]. The larger FP16 gathered-input
    // view [C*8][kE] is needed solely by the non-MMQ fallback, which the compact-mode guard excludes.
    P.Xe = hal(P.compact_workspace ? (size_t)C * 8 * kFF * 2 : (size_t)C * 8 * kE);
    P.act = hal((size_t)C * kFF);
    if (!P.compact_workspace) {
        P.Wgu = hal((size_t)2 * kFF * kE);
        P.Wdn = hal((size_t)kE * kFF);
        P.Qh = hal((size_t)C * kNH * kHD);
    }
    P.ids = ial((size_t)C * 8);
    P.perm = ial((size_t)C * 16);
    P.inv = P.perm + (size_t)C * 8;
#if defined(SQ_PREFILL_MMQ)
    if (P.use_mmq) {
        const size_t xq_bytes = strata::prefill::mmq::q8_bytes((int64_t)C * 8, kE);
        // A 16-expert group may contain every routed row (8 assignments/token),
        // so its compact down activation is sized for C*8, not C.
        const size_t hq_bytes = strata::prefill::mmq::q8_bytes((int64_t)C * 8, kFF);
        for (const LayerW& L : e_.model_.layers) {
            P.mmq_gu_stride = std::max(P.mmq_gu_stride, (size_t)2 * L.gu_bytes);
            P.mmq_down_stride = std::max(P.mmq_down_stride, (size_t)L.down_bytes);
        }
        P.xq_stride = strata::prefill::mmq::q8_bytes(2, kE) - strata::prefill::mmq::q8_bytes(1, kE);
        P.hq_stride = strata::prefill::mmq::q8_bytes(2, kFF) - strata::prefill::mmq::q8_bytes(1, kFF);
        CUDA_CHECK(cudaMalloc(&P.Xq, xq_bytes));
        CUDA_CHECK(cudaMalloc(&P.Hq, hq_bytes));
        CUDA_CHECK(cudaMalloc(&P.mmq_gu, kMMQGroup * P.mmq_gu_stride + kMMQTail));
        CUDA_CHECK(cudaMalloc(&P.mmq_down, kMMQGroup * P.mmq_down_stride + kMMQTail));
        if (P.group_prefetch) {
            CUDA_CHECK(cudaMalloc(&P.mmq_gu_alt, kMMQGroup * P.mmq_gu_stride + kMMQTail));
            CUDA_CHECK(cudaMalloc(&P.mmq_down_alt, kMMQGroup * P.mmq_down_stride + kMMQTail));
            for (int i = 0; i < 2; ++i) {
                CUDA_CHECK(cudaEventCreateWithFlags(&P.mmq_ready[i], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(&P.mmq_free[i], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(P.mmq_free[i], P.s));
            }
            log("layer-major group prefetch: two native MMQ weight banks, %.1f MiB extra; direct H2D/D2D on copy stream",
                (kMMQGroup * (P.mmq_gu_stride + P.mmq_down_stride) + 2 * kMMQTail) / 1048576.0);
        }
        CUDA_CHECK(cudaMalloc(&P.mmq_identity, (size_t)C * 8 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&P.mmq_bounds, P.mmq_bounds_capacity * sizeof(int)));
        CUDA_CHECK(cudaHostAlloc((void**)&P.h_mmq_bounds, P.mmq_bounds_capacity * sizeof(int), cudaHostAllocDefault));
        P.mmq = std::make_unique<strata::prefill::mmq::Context>();
        strata::prefill::mmq::iota(P.mmq_identity, (int64_t)C * 8, P.s);
        const int mmq_layers = (int)std::count(P.mmq_layer.begin(), P.mmq_layer.end(), (char)1);
        log("prefill packed MMQ: Pascal DP4A on %d/%zu layers (%.1f MiB q8 activations, %.1f MiB 16-expert weights)",
            mmq_layers, P.mmq_layer.size(),
            (xq_bytes + hq_bytes) / 1048576.0,
            (kMMQGroup * (P.mmq_gu_stride + P.mmq_down_stride) + 2 * kMMQTail) / 1048576.0);
    }
#endif
    CUDA_CHECK(cudaHostAlloc((void**)&P.h_ids, (size_t)C * 8 * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&P.h_perm, (size_t)C * 16 * 4, cudaHostAllocDefault));
    for (const LayerW& L : e_.model_.layers) P.stage_bytes = std::max(P.stage_bytes, L.blob_bytes);
    const size_t stage_capacity = (size_t)P.stage_bytes * kStage;
#if defined(SQ_PREFILL_MMQ)
    if (P.group_prefetch && P.mmq_gu_alt && kMMQGroup * P.mmq_gu_stride + kMMQTail >= stage_capacity) {
        // Main layers in group-prefetch mode never use the staging ring. Ordinary MTP begins after the last
        // main-layer stream synchronization and uses only the primary MMQ bank, so its ring can occupy the
        // alternate GU bank. MTP synchronizes all ring consumers before the next window reuses that bank.
        P.stage = P.mmq_gu_alt;
        P.stage_alias = true;
        log("layer-major group prefetch: MTP staging reuses %.1f MiB of alternate GU bank; no separate ring allocation",
            stage_capacity / 1048576.0);
    } else if (P.compact_workspace && stage_capacity <= big_cols * kE * sizeof(__half)) {
        // W16's last dense/shared GEMM is complete at the existing shared-expert stream synchronization,
        // before any ring copy is submitted. All-MMQ MoE reads only the separate native weight banks.
        // Every queued ring copy is waited on and gathered on P.s; the next W16 write follows those
        // consumers on P.s (main layers also synchronize at their end). Ordinary MTP has the same order.
        // Keep the alternate-bank alias above as first choice for group-prefetch mode.
        P.stage = reinterpret_cast<uint8_t*>(P.W16);
        P.stage_alias = true;
        log("compact prefill workspace: staging reuses %.2f MiB of idle dense-matrix workspace",
            stage_capacity / 1048576.0);
    } else
#endif
    {
        CUDA_CHECK(cudaMalloc(&P.stage, stage_capacity));
    }
    for (int i = 0; i < kStage; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&P.ev_copied[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&P.ev_free[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(P.ev_free[i], P.s));
    }
    const size_t shared_limit = std::max(prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin);
    P.small_attn = shared_limit < attn_smem<8>();
    if (const char* v = std::getenv("STRATA_PREFILL_ATTN_KV_SHARED"))
        P.shared_attn_kv = *v && std::strcmp(v, "0") != 0;
    if (P.shared_attn_kv) {
        P.attn_aq = 4;
        if (const char* v = std::getenv("STRATA_PREFILL_ATTN_AQ")) P.attn_aq = std::atoi(v);
        switch (P.attn_aq) {
            case 2: configure_shared_attn<2>(prop, shared_limit); break;
            case 3: configure_shared_attn<3>(prop, shared_limit); break;
            case 4: configure_shared_attn<4>(prop, shared_limit); break;
            case 6: configure_shared_attn<6>(prop, shared_limit); break;
            default: SQ_CHECK(false, "STRATA_PREFILL_ATTN_AQ must be 2, 3, 4 or 6 when shared K/V is enabled");
        }
    } else {
        P.attn_aq = P.small_attn ? 2 : 8;
        SQ_CHECK(shared_limit >= attn_smem<2>(), "prompt attention needs at least %zu bytes of shared memory", attn_smem<2>());
        if (!P.small_attn) {
            CUDA_CHECK(cudaFuncSetAttribute(attn_prefill_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int)attn_smem<8>()));
        }
        if (P.small_attn) log("prompt attention: 2-token tile (%zu bytes shared memory) for %s", attn_smem<2>(), prop.name);
    }
    if (const char* v = std::getenv("STRATA_PREFILL_ATTN_BANK_LAYOUT"); v && std::strcmp(v, "1") == 0) {
        P.attn_bank_layout = P.layer_major && P.shared_attn_kv && P.attn_aq == 6;
        if (P.attn_bank_layout) {
            constexpr size_t bytes = attn_layout_smem<6, true, true, true>();
            static_assert(bytes == 48960, "AQ6 bank-layout shared memory changed");
            SQ_CHECK(192 <= prop.maxThreadsPerBlock && 192 <= prop.maxThreadsDim[0],
                     "AQ6 bank-layout attention needs 192 threads per block");
            SQ_CHECK(bytes <= shared_limit, "AQ6 bank-layout attention needs %zu bytes, device supports %zu", bytes, shared_limit);
            if (bytes > prop.sharedMemPerBlock)
                CUDA_CHECK(cudaFuncSetAttribute(attn_layout_kernel<6, true, true, true>,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
            log("prompt attention: AQ6 bank layout, %zu bytes shared memory (+768), no additional global VRAM; prepared main layers only", bytes);
        } else {
            log("prompt attention bank layout not enabled: requires layer-major prefill with shared K/V and AQ6");
        }
    }
    size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
    if (P.dense_f32) {
        SQ_CHECK(kE <= kConv, "dense FP32 activation scratch exceeds the convolution allocation");
        log("experimental dense FP32 prefill: prepared main-layer input projections only; %.2f MiB extra matrix storage; "
            "Q8 weights retain half rounding, activations widen exactly into reused conv scratch; "
            "SGEMM reduction order may change results; mixer input timings include activation widening",
            big_cols * kE * sizeof(__half) / 1048576.0);
    }
    if (P.compact_workspace) {
        const size_t gu_saved = (size_t)C * 8 * 2 * kFF * sizeof(float);
        const size_t qh_saved = (size_t)C * kNH * kHD * sizeof(__half);
        const size_t xe_saved = (size_t)C * 8 * (kE * sizeof(__half) - kFF * sizeof(float));
        const size_t fallback_saved = (size_t)3 * kFF * kE * sizeof(__half);
        const size_t stage_saved = P.stage == reinterpret_cast<uint8_t*>(P.W16) ? stage_capacity : 0;
        log("compact prefill workspace: %.2f MiB allocation saved (gu/Qh aliases %.2f MiB, MMQ H sizing %.2f MiB, "
            "unused expert matrices %.2f MiB, staging alias %.2f MiB); every main/MTP layer uses MMQ",
            (gu_saved + qh_saved + xe_saved + fallback_saved + stage_saved) / 1048576.0,
            (gu_saved + qh_saved) / 1048576.0, xe_saved / 1048576.0, fallback_saved / 1048576.0,
            stage_saved / 1048576.0);
    }
    log("prefill workspace: %s %d, %.1f MiB device allocation, %.1f MiB CUDA free while active",
        P.layer_major ? "layer-major tile" : "chunk", P.layer_major ? P.layer_tile : e_.opt_.prefill_chunk,
        (free_before - free_after) / 1048576.0, free_after / 1048576.0);
    ready_ = true;
    return true;
}

namespace {

/// Y[T][rows] (fp32) = X[T][cols] (fp16) * W[rows][cols]^T (fp16); beta 1 accumulates into Y.
void gemm(cublasHandle_t cb, const __half* W, const __half* X, float* Y, int rows, int cols, int T, int ldy, float beta) {
    const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, cols, &alpha, W, CUDA_R_16F, cols, X, CUDA_R_16F, cols,
                              &beta, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_f32(cublasHandle_t cb, const float* W, const float* X, float* Y, int rows, int cols, int T, int ldy,
              float beta = 0.0f) {
    const float alpha = 1.0f;
    CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, cols, &alpha, W, cols, X, cols, &beta, Y, ldy));
}

void dequant_q8(const DQ8& W, __half* out, cudaStream_t s) {
    const int64_t n = (int64_t)W.rows * W.cols;
    dequant_q8_kernel<<<(unsigned)((n / 8 + 255) / 256), 256, 0, s>>>(W.qs, (const __half*)W.d, out, n);
}

void dequant_q8_half_f32(const DQ8& W, __half* storage, cudaStream_t s) {
    const int64_t n = (int64_t)W.rows * W.cols;
    dequant_q8_half_f32_kernel<<<(unsigned)((n / 8 + 255) / 256), 256, 0, s>>>(
        W.qs, (const __half*)W.d, reinterpret_cast<float*>(storage), n);
}

// Opt-in numerical diagnostics only. Files are headerless row-major FP32 (hidden) or INT32 (router IDs),
// with up to the first 1024 prompt rows. The caller supplies an existing output directory.
void diagnostic_write(const std::string& dir, const char* mode, int il, const char* stage,
                      const void* data, size_t bytes) {
    char suffix[96];
    std::snprintf(suffix, sizeof(suffix), "/%s-layer%02d-%s.bin", mode, il, stage);
    const std::string path = dir + suffix;
    FILE* f = std::fopen(path.c_str(), "wb");
    SQ_CHECK(f, "cannot open prefill diagnostic %s", path.c_str());
    const size_t written = std::fwrite(data, 1, bytes, f);
    const int closed = std::fclose(f);
    SQ_CHECK(written == bytes && closed == 0, "cannot write prefill diagnostic %s", path.c_str());
}

void diagnostic_dump(const std::string& dir, const char* mode, int il, const char* stage,
                     const void* device, int rows, int cols, cudaStream_t s) {
    const size_t bytes = (size_t)std::min(rows, 1024) * cols * sizeof(uint32_t);
    std::vector<uint32_t> host(bytes / sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpyAsync(host.data(), device, bytes, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    diagnostic_write(dir, mode, il, stage, host.data(), bytes);
}

}  // namespace

void Prefill::mixer_prepared(int il, int T, int p0) {
    Impl& P = *p_;
    Engine& E = e_;
    const Config& c = E.model_.cfg;
    const LayerW& L = E.model_.layers[il];
    const float eps = c.eps;
    cudaStream_t s = P.s;
    auto mark = [&](MixerProfile::Point point) {
        if (P.mixer_profile) P.mixer_profile->mark(P.mixer_profile_tile, point, s);
    };
    auto dense_gemm = [&](const __half* W, const __half* X, float* Y, int rows, int cols, int ldy, float beta) {
        if (P.dense_f32) {
            SQ_CHECK(T > 0 && T <= P.C && cols > 0 && cols <= kConv,
                     "dense FP32 activation scratch exceeds the convolution allocation");
            // Before the input projection conv (including its compact Qh/gu aliases) is dead.
            // Its next use follows this GEMM on s.
            // Preserve Hn's unrounded FP32 rows for the independent GDN BA projection.
            const int64_t n = (int64_t)T * cols;
            widen_half_kernel<<<(unsigned)((n + 255) / 256), 256, 0, s>>>(X, P.conv, n);
#if !defined(__HIPCC__) && !defined(GGML_USE_HIP)
            // Only the measured full-tile prepared input projections are eligible.
            // BA/router, FP16 output projections, ordinary MTP, and decoding keep their paths.
            static_assert(kE == 2048, "algorithm 6 input-width evidence must be revalidated");
            static_assert(static_cast<int>(CUBLAS_GEMM_ALGO6) == 6, "cuBLAS algorithm enum changed");
            if (P.dense_algo6 && il >= 0 && il < c.n_layer && T == 1024 && cols == 2048 &&
                (rows == 12288 || rows == 9216) && ldy == rows && beta == 0.0f) {
                const float alpha = 1.0f;
                const cublasStatus_t status = cublasGemmEx(
                    P.cb, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, cols, &alpha,
                    reinterpret_cast<const float*>(W), CUDA_R_32F, cols,
                    P.conv, CUDA_R_32F, cols, &beta, Y, CUDA_R_32F, ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_ALGO6);
                if (status == CUBLAS_STATUS_SUCCESS) return;
                if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) {
                    // Same-stream beta0 SGEMM overwrites this result. Disable further attempts
                    // for this instance, avoiding repeated API failures/logging on unsupported drivers.
                    P.dense_algo6 = false;
                    log("prompt FP32 algorithm 6 disabled after unsupported cuBLAS status %d; using SGEMM", (int)status);
                } else {
                    CUBLAS_CHECK(status);
                }
            }
#endif
            gemm_f32(P.cb, reinterpret_cast<const float*>(W), P.conv, Y, rows, cols, T, ldy, beta);
        } else {
            gemm(P.cb, W, X, Y, rows, cols, T, ldy, beta);
        }
    };
    SQ_CHECK(P.W16b, "layer-major output matrix buffer is missing");
    rmsnorm_rows_kernel<<<T, 256, 0, s>>>(P.X, L.attn_norm, eps, L.attn ? nullptr : P.Hn, P.Hh);
    mark(MixerProfile::RMS);
    if (!L.attn) {
        const int ld = L.qkvz.rows;
        dense_gemm(P.W16, P.Hh, P.big, L.qkvz.rows, kE, ld, 0.0f);
        mark(MixerProfile::INPUT);
        gemm_f32(P.cb, L.ba.w, P.Hn, P.ba, 64, kE, T, 64);
        mark(MixerProfile::BA);
        conv_seq_kernel<<<kConv / 256, 256, 0, s>>>(P.big, ld, E.conv_st_[il], L.conv_w, P.conv, T);
        mark(MixerProfile::CONV);
        gdn_prep_kernel<<<T, 1024, 0, s>>>(P.conv, P.ba, L.dt_bias, L.ssm_a, P.gb);
        mark(MixerProfile::GDN_PREP);
        gdn_scan(P.conv, P.gb, E.ssm_st_[il], P.O, T, P.gdn_columns, s);
        mark(MixerProfile::GDN_SCAN);
        gdn_normgate_rows_kernel<<<dim3(T, kVH), kDS, 0, s>>>(P.O, P.big, ld, L.ssm_norm, eps, P.Y);
        mark(MixerProfile::GDN_NORM);
        gemm(P.cb, P.W16b, P.Y, P.X, kE, 4096, T, kE, 1.0f);
        mark(MixerProfile::OUTPUT);
    } else {
        const int ld = L.qkv.rows;
        dense_gemm(P.W16, P.Hh, P.big, L.qkv.rows, kE, ld, 0.0f);
        mark(MixerProfile::INPUT);
        attn_prep_rows_kernel<<<dim3(T, kNH + 2 * kNKV), 256, 0, s>>>(P.big, ld, L.q_norm, L.k_norm, eps, p0,
                                                                      c.rope_base, P.Qh, (__half*)E.kc_[il],
                                                                      (__half*)E.vc_[il], E.kv_capacity_);
        mark(MixerProfile::ATTN_PREP);
        if (P.attn_bank_layout && il < c.n_layer) {
            attn_layout_kernel<6, true, true, true><<<dim3((T + 5) / 6, kNKV), 192,
                                                     attn_layout_smem<6, true, true, true>(), s>>>(
                P.Qh, (const __half*)E.kc_[il], (const __half*)E.vc_[il], T, p0, E.kv_capacity_, P.big, ld, P.Y);
        } else {
            launch_attn_prefill(P.shared_attn_kv, P.attn_aq, P.Qh, (const __half*)E.kc_[il], (const __half*)E.vc_[il],
                                T, p0, E.kv_capacity_, P.big, ld, P.Y, s);
        }
        mark(MixerProfile::ATTN);
        gemm(P.cb, P.W16b, P.Y, P.X, kE, 4096, T, kE, 1.0f);
        mark(MixerProfile::OUTPUT);
    }
}

void Prefill::layer(int il, int T, int p0) {
    Impl& P = *p_;
    Engine& E = e_;
    const Config& c = E.model_.cfg;
    const float eps = c.eps;
    cudaStream_t s = P.s;
    const LayerW& L = E.model_.layers[il];
    const bool diagnostic = !P.diagnostic_dir.empty() && p0 == 0 && il < c.n_layer &&
                            (P.diagnostic_layer < 0 || P.diagnostic_layer == il);
#if defined(SQ_PREFILL_MMQ)
    const bool use_mmq_layer = P.use_mmq && il >= 0 && (size_t)il < P.mmq_layer.size() && P.mmq_layer[(size_t)il];
#endif
    rmsnorm_rows_kernel<<<T, 256, 0, s>>>(P.X, L.attn_norm, eps, L.attn ? nullptr : P.Hn, P.Hh);
    if (!L.attn) {
        const int ld = L.qkvz.rows;   // 12288
        dequant_q8(L.qkvz, P.W16, s);
        gemm(P.cb, P.W16, P.Hh, P.big, L.qkvz.rows, kE, T, ld, 0.0f);
        gemm_f32(P.cb, L.ba.w, P.Hn, P.ba, 64, kE, T, 64);
        conv_seq_kernel<<<kConv / 256, 256, 0, s>>>(P.big, ld, E.conv_st_[il], L.conv_w, P.conv, T);
        gdn_prep_kernel<<<T, 1024, 0, s>>>(P.conv, P.ba, L.dt_bias, L.ssm_a, P.gb);
        gdn_scan(P.conv, P.gb, E.ssm_st_[il], P.O, T, P.gdn_columns, s);
        gdn_normgate_rows_kernel<<<dim3(T, kVH), kDS, 0, s>>>(P.O, P.big, ld, L.ssm_norm, eps, P.Y);
        dequant_q8(L.ssm_out, P.W16, s);
        gemm(P.cb, P.W16, P.Y, P.X, kE, 4096, T, kE, 1.0f);
    } else {
        const int ld = L.qkv.rows;    // 9216
        dequant_q8(L.qkv, P.W16, s);
        gemm(P.cb, P.W16, P.Hh, P.big, L.qkv.rows, kE, T, ld, 0.0f);
        attn_prep_rows_kernel<<<dim3(T, kNH + 2 * kNKV), 256, 0, s>>>(P.big, ld, L.q_norm, L.k_norm, eps, p0, c.rope_base,
                                                                      P.Qh, (__half*)E.kc_[il], (__half*)E.vc_[il], E.kv_capacity_);
        launch_attn_prefill(P.shared_attn_kv, P.attn_aq, P.Qh, (const __half*)E.kc_[il], (const __half*)E.vc_[il],
                            T, p0, E.kv_capacity_, P.big, ld, P.Y, s);
        dequant_q8(L.wo, P.W16, s);
        gemm(P.cb, P.W16, P.Y, P.X, kE, 4096, T, kE, 1.0f);
    }
    if (diagnostic) diagnostic_dump(P.diagnostic_dir, "chunk", il, "postmixer", P.X, T, kE, s);
    // ---- MoE
    rmsnorm_rows_kernel<<<T, 256, 0, s>>>(P.X, L.post_norm, eps, P.Hn, P.Hh);
    gemm_f32(P.cb, L.router.w, P.Hn, P.rlog, 257, kE, T, 257);
    router_rows_kernel<<<T, 256, 0, s>>>(P.rlog, 257, P.ids, P.wts, E.route_counts_, il);
    if (diagnostic) diagnostic_dump(P.diagnostic_dir, "chunk", il, "routerids", P.ids, T, 8, s);
    CUDA_CHECK(cudaMemcpyAsync(P.h_ids, P.ids, (size_t)T * 8 * 4, cudaMemcpyDeviceToHost, s));
    // shared expert while the host waits for the routing
    dequant_q8(L.sh_gu, P.W16, s);
    gemm(P.cb, P.W16, P.Hh, P.shgu, 2 * kFF, kE, T, 2 * kFF, 0.0f);
    swiglu_rows_kernel<<<T, 256, 0, s>>>(P.shgu, P.act, kFF);
    dequant_q8(L.sh_down, P.W16, s);
    gemm(P.cb, P.W16, P.act, P.shout, kE, kFF, T, kE, 0.0f);
    CUDA_CHECK(cudaStreamSynchronize(s));
    // group assignments by expert (counting sort)
    int cnt[257] = {0};
    for (int a = 0; a < T * 8; ++a) ++cnt[P.h_ids[a] + 1];
    for (int e = 0; e < 256; ++e) cnt[e + 1] += cnt[e];
    int off[256];
    std::memcpy(off, cnt, sizeof(off));
    int* perm = P.h_perm;
    int* inv = P.h_perm + (size_t)T * 8;
    for (int a = 0; a < T * 8; ++a) {
        const int pos = off[P.h_ids[a]]++;
        perm[pos] = a >> 3;   // token of this assignment
        inv[a] = pos;
    }
    CUDA_CHECK(cudaMemcpyAsync(P.perm, perm, (size_t)T * 8 * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(P.inv, inv, (size_t)T * 8 * 4, cudaMemcpyHostToDevice, s));
#if defined(SQ_PREFILL_MMQ)
    if (use_mmq_layer)
        strata::prefill::mmq::quantize(P.Hn, P.perm, P.Xq, (int)L.t_gu, kE, kE, (int64_t)T * 8, s);
    else
#endif
        gather_rows_kernel<<<T * 8, 256, 0, s>>>(P.Hh, P.perm, P.Xe);
    // stream non-resident experts through the staging ring, a few ahead of the compute
    std::vector<int> order;
    for (int e = 0; e < 256; ++e)
        if (cnt[e + 1] > cnt[e]) order.push_back(e);
    std::vector<int> stage_of(order.size(), -1);
    size_t next_copy = 0;
    int ring = 0;
    auto issue_copies = [&](size_t upto) {
        for (; next_copy < order.size() && next_copy < upto; ++next_copy) {
            const int e = order[next_copy];
            if (E.residency_host_[(size_t)il * 256 + e] >= 0) continue;
            const int sl = ring++ % kStage;
            stage_of[next_copy] = sl;
            CUDA_CHECK(cudaStreamWaitEvent(P.cs, P.ev_free[sl], 0));
            CUDA_CHECK(cudaMemcpyAsync(P.stage + (size_t)sl * P.stage_bytes, L.expert_blob(e), (size_t)L.blob_bytes,
                                       cudaMemcpyHostToDevice, P.cs));
            CUDA_CHECK(cudaEventRecord(P.ev_copied[sl], P.cs));
        }
    };
    // how many experts ahead may be in flight: limited by the ring
    auto ahead_limit = [&](size_t i) {
        size_t k = i, used = 0;
        while (k < order.size() && used < (size_t)kStage) {
            if (E.residency_host_[(size_t)il * 256 + order[k]] < 0) ++used;
            ++k;
        }
        return k;
    };
#if defined(SQ_PREFILL_MMQ)
    if (use_mmq_layer) {
        // One expert per MMQ launch loses the packed arithmetic win to launch overhead on Pascal.
        // Gather up to 16 native blobs, then run one GU and one down product for the group.
        const size_t n = order.size(), ng = (n + kMMQGroup - 1) / kMMQGroup;
        const size_t layer_gu_stride = (size_t)2 * L.gu_bytes;
        const size_t layer_down_stride = (size_t)L.down_bytes;
        for (size_t j = 0; j < n; ++j) P.h_mmq_bounds[j] = cnt[order[j]];
        P.h_mmq_bounds[n] = T * 8;
        for (size_t g = 0; g < ng; ++g) {
            const size_t j0 = g * kMMQGroup;
            for (int q = 0; q <= kMMQGroup; ++q) {
                const size_t j = std::min(n, j0 + (size_t)q);
                P.h_mmq_bounds[n + 1 + g * (kMMQGroup + 1) + q] =
                    P.h_mmq_bounds[j] - P.h_mmq_bounds[j0];
            }
        }
        const size_t nbounds = n + 1 + ng * (kMMQGroup + 1);
        CUDA_CHECK(cudaMemcpyAsync(P.mmq_bounds, P.h_mmq_bounds, nbounds * sizeof(int), cudaMemcpyHostToDevice, s));

        float* mmq_h = reinterpret_cast<float*>(P.Xe); // fallback Xe is idle and large enough for routed FP32 H.
        for (size_t g = 0; g < ng; ++g) {
            const size_t j0 = g * kMMQGroup, j1 = std::min(n, j0 + kMMQGroup);
            const int ngx = (int)(j1 - j0);
            int max_rows = 0;
            for (size_t j = j0; j < j1; ++j) {
                issue_copies(ahead_limit(j));
                const int e = order[j];
                const int slot = E.residency_host_[(size_t)il * 256 + e];
                const uint8_t* blob;
                if (slot >= 0) {
                    blob = E.slot_base_[il] + (size_t)slot * L.blob_bytes;
                } else {
                    const int sl = stage_of[j];
                    CUDA_CHECK(cudaStreamWaitEvent(s, P.ev_copied[sl], 0));
                    blob = P.stage + (size_t)sl * P.stage_bytes;
                }
                const size_t q = j - j0;
                strata::prefill::mmq::gather_native(blob, blob + L.gu_bytes, (size_t)L.gu_bytes,
                    blob + 2 * L.gu_bytes, (size_t)L.down_bytes,
                    P.mmq_gu + q * layer_gu_stride, P.mmq_down + q * layer_down_stride, s);
                if (slot < 0) CUDA_CHECK(cudaEventRecord(P.ev_free[stage_of[j]], s));
                max_rows = std::max(max_rows, cnt[e + 1] - cnt[e]);
            }
            CUDA_CHECK(cudaMemsetAsync(P.mmq_gu + (size_t)ngx * layer_gu_stride, 0, kMMQTail, s));
            CUDA_CHECK(cudaMemsetAsync(P.mmq_down + (size_t)ngx * layer_down_stride, 0, kMMQTail, s));

            const int r0 = P.h_mmq_bounds[j0], nr = P.h_mmq_bounds[j1] - r0;
            strata::prefill::mmq::Product gu{};
            gu.w = P.mmq_gu;
            gu.type = (int)L.t_gu;
            gu.w_rows = 2 * kFF;
            gu.w_cols = kE;
            gu.expert_bytes = layer_gu_stride;
            gu.n = ngx;
            gu.xq = P.Xq;
            gu.bounds = P.mmq_bounds + j0;
            gu.ids = P.mmq_identity;
            gu.total_rows = (int64_t)T * 8;
            gu.max_rows = max_rows;
            gu.dst = P.gu;
            gu.ld_dst = 2 * kFF;
            P.mmq->run(gu, s);
            strata::prefill::mmq::swiglu(P.gu + (size_t)r0 * 2 * kFF,
                                         mmq_h + (size_t)r0 * kFF, nr, kFF, false, s);
            strata::prefill::mmq::quantize(mmq_h + (size_t)r0 * kFF, nullptr, P.Hq,
                                           (int)L.t_down, kFF, kFF, nr, s);

            strata::prefill::mmq::Product down{};
            down.w = P.mmq_down;
            down.type = (int)L.t_down;
            down.w_rows = kE;
            down.w_cols = kFF;
            down.expert_bytes = layer_down_stride;
            down.n = ngx;
            down.xq = P.Hq;
            down.bounds = P.mmq_bounds + n + 1 + g * (kMMQGroup + 1);
            down.ids = P.mmq_identity;
            down.total_rows = nr;
            down.max_rows = max_rows;
            down.dst = P.moe + (size_t)r0 * kE;
            down.ld_dst = kE;
            P.mmq->run(down, s);
        }
    } else
#endif
    {
        for (size_t i = 0; i < order.size(); ++i) {
            issue_copies(ahead_limit(i));
            const int e = order[i];
            const int ne = cnt[e + 1] - cnt[e], o0 = cnt[e];
            const uint8_t* blob;
            const int slot = E.residency_host_[(size_t)il * 256 + e];
            if (slot >= 0) {
                blob = E.slot_base_[il] + (size_t)slot * L.blob_bytes;
            } else {
                const int sl = stage_of[i];
                CUDA_CHECK(cudaStreamWaitEvent(s, P.ev_copied[sl], 0));
                blob = P.stage + (size_t)sl * P.stage_bytes;
            }
            dequant_native_expert(L.t_gu, blob, P.Wgu, (int64_t)2 * kFF * kE, s);
            dequant_native_expert(L.t_down, blob + 2 * L.gu_bytes, P.Wdn, (int64_t)kE * kFF, s);
            if (slot < 0) CUDA_CHECK(cudaEventRecord(P.ev_free[stage_of[i]], s));
            gemm(P.cb, P.Wgu, P.Xe + (size_t)o0 * kE, P.gu, 2 * kFF, kE, ne, 2 * kFF, 0.0f);
            swiglu_rows_kernel<<<ne, 256, 0, s>>>(P.gu, P.act, kFF);
            gemm(P.cb, P.Wdn, P.act, P.moe + (size_t)o0 * kE, kE, kFF, ne, kE, 0.0f);
        }
    }
    moe_combine_rows_kernel<<<T, 256, 0, s>>>(P.X, P.moe, P.inv, P.wts, P.shout, P.rlog, 257);
    if (diagnostic) diagnostic_dump(P.diagnostic_dir, "chunk", il, "postmoe", P.X, T, kE, s);
}

void Prefill::mtp(const int* tokens, int n, int done, int T, int p0) {
    Impl& P = *p_;
    Engine& E = e_;
    const Config& c = E.model_.cfg;
    const MtpW& M = E.model_.mtp;
    cudaStream_t s = P.s;
    const int k = E.mtp_k_;                  // pending pairs (positions p0-k .. p0-1), hiddens in mtp_hin_
    const bool last = done + T == n;
    const int m = last ? T - 1 : T;          // chunk rows whose next token is known
    const int Tm = k + m, q = p0 - k;
    // the trunk's normed final hidden states of the chunk
    rmsnorm_rows_kernel<<<T, 256, 0, s>>>(P.X, E.model_.output_norm, c.eps, P.Hn, P.Hh);
    float* MH = P.big;                       // [Tm][2048] pair hiddens (P.big is free until the layer's projections)
    if (k > 0) CUDA_CHECK(cudaMemcpyAsync(MH, E.mtp_hin_, (size_t)k * kE * 4, cudaMemcpyDeviceToDevice, s));
    if (m > 0) CUDA_CHECK(cudaMemcpyAsync(MH + (size_t)k * kE, P.Hn, (size_t)m * kE * 4, cudaMemcpyDeviceToDevice, s));
    if (last) CUDA_CHECK(cudaMemcpyAsync(E.mtp_hin_, P.Hn + (size_t)(T - 1) * kE, (size_t)kE * 4, cudaMemcpyDeviceToDevice, s));
    E.mtp_k_ = last ? 1 : 0;
    if (Tm == 0) return;
    // embeddings of the pairs' next tokens (position q + r + 1)
    const int base = E.n_past_;              // tokens[0] is at position base
    std::vector<float> emb((size_t)Tm * kE);
    for (int r = 0; r < Tm; ++r) {
        const int g = q + r + 1;
        const int t = g < base ? E.history_[g] : tokens[g - base];
        E.model_.embed(t, emb.data() + (size_t)r * kE);
    }
    float* EM = P.shout;                     // free until the layer's shared expert
    CUDA_CHECK(cudaMemcpyAsync(EM, emb.data(), emb.size() * 4, cudaMemcpyHostToDevice, s));
    rmsnorm_rows_ld_kernel<<<Tm, 256, 0, s>>>(EM, M.enorm, c.eps, P.Y, 2 * kE);
    rmsnorm_rows_ld_kernel<<<Tm, 256, 0, s>>>(MH, M.hnorm, c.eps, P.Y + kE, 2 * kE);
    dequant_q8(M.eh_proj, P.W16, s);
    gemm(P.cb, P.W16, P.Y, P.X, kE, 2 * kE, Tm, kE, 0.0f);
    CUDA_CHECK(cudaStreamSynchronize(s));    // emb (pageable) is released below
    layer(c.n_layer, Tm, q);
}

void Prefill::run_layer_major(const int* tokens, int n) {
    Impl& P = *p_;
    const char* ordered_env = std::getenv("STRATA_PREFILL_LAYER_ORDERED_COMBINE");
    const bool ordered_combine = !ordered_env || std::strcmp(ordered_env, "0") != 0;
    int requested_window = (8192 / P.layer_tile) * P.layer_tile;
    if (const char* v = std::getenv("STRATA_PREFILL_LAYER_WINDOW")) requested_window = std::atoi(v);
    SQ_CHECK(requested_window > 0 && requested_window % P.layer_tile == 0,
             "STRATA_PREFILL_LAYER_WINDOW must be a positive multiple of layer tile %d", P.layer_tile);
    if (!ordered_combine)
        log("layer-major EXPERIMENTAL compact combine: router-order/FMA parity is not preserved");

    // Bound prompt-wide state independently of the requested context. The original combine kernel needs
    // all eight routed outputs, so memory grows by 91,236 bytes/token rather than retaining the whole prompt.
    // Free VRAM is sampled after tile workspace allocation and again between windows to accommodate WDDM.
    const size_t bytes_per_token = 3 * kE * sizeof(float) + 8 * (sizeof(float) + 2 * sizeof(int)) +
                                   (ordered_combine ? (8 * kE + 257) * sizeof(float) : 0);
    constexpr size_t reserve = (size_t)128 * 1048576;
    for (int done = 0; done < n;) {
        size_t free_now = 0, total_now = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_now, &total_now));
        const size_t fits = free_now > reserve ? (free_now - reserve) / bytes_per_token : 0;
        const int remaining = n - done;
        int count = (int)std::min<size_t>(std::min(remaining, requested_window), fits);
        // Keep normal windows aligned to the configured tile. Under tighter budgets, 128-row windows remain
        // valid with the same allocated workspace. The final short tail need not be a multiple of either size.
        if (count < remaining) {
            const int align = count >= P.layer_tile ? P.layer_tile : 128;
            count = count / align * align;
        }
        SQ_CHECK(count > 0,
                 "layer-major needs space for at least %d rows plus 128 MiB reserve; CUDA free %.1f MiB",
                 std::min(remaining, 128), free_now / 1048576.0);
        log("layer-major window: offset %d, %d/%d remaining tokens, cap %d, tile %d, %.1f MiB state, %.1f MiB CUDA free before state, 128 MiB reserve",
            done, count, remaining, requested_window, P.layer_tile,
            (size_t)count * bytes_per_token / 1048576.0, free_now / 1048576.0);
        run_layer_major_window(tokens, n, done, count, ordered_combine);
        done += count;
    }
}

void Prefill::run_layer_major_window(const int* tokens, int total_n, int window_start, int n, bool ordered_combine) {
#if !defined(SQ_PREFILL_MMQ)
    (void)tokens; (void)total_n; (void)window_start; (void)n; (void)ordered_combine;
    SQ_CHECK(false, "layer-major prefill requires packed MMQ support");
#else
    Impl& P = *p_;
    Engine& E = e_;
    const Config& c = E.model_.cfg;
    cudaStream_t s = P.s;
    SQ_CHECK(P.use_mmq && std::all_of(P.mmq_layer.begin(), P.mmq_layer.end(), [](char x) { return x != 0; }),
             "layer-major prefill requires MMQ coverage for every main and MTP layer");
    const int tile = P.layer_tile;
    int route_cap = tile * 8;
    // A lower cap is useful for exercising compact row subpasses with short prompts. The default preserves
    // the ordinary one-tile global-row layout; a smaller override can deliberately select the compact path.
    if (const char* v = std::getenv("STRATA_PREFILL_LAYER_ROUTE_CAP")) {
        const int requested = std::atoi(v);
        SQ_CHECK(requested > 0 && requested <= route_cap,
                 "STRATA_PREFILL_LAYER_ROUTE_CAP must be between 1 and %d", route_cap);
        route_cap = requested;
    }
    float *full_x = nullptr, *full_hn = nullptr, *full_acc = nullptr, *full_w = nullptr, *full_moe = nullptr;
    float* full_rlog = nullptr;
    int *d_perm = nullptr, *d_inv = nullptr;
    int* h_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&full_x, (size_t)n * kE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&full_hn, (size_t)n * kE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&full_acc, (size_t)n * kE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&full_w, (size_t)n * 8 * sizeof(float)));
    if (ordered_combine) {
        const size_t bytes = (size_t)n * 8 * kE * sizeof(float);
        CUDA_CHECK(cudaMalloc(&full_moe, bytes));
        CUDA_CHECK(cudaMalloc(&full_rlog, (size_t)n * 257 * sizeof(float)));
        log("layer-major ordered combine: %.1f MiB window routed outputs + %.1f MiB router logits; original combine kernel",
            bytes / 1048576.0, (size_t)n * 257 * sizeof(float) / 1048576.0);
    }
    CUDA_CHECK(cudaMalloc(&d_perm, (size_t)n * 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inv, (size_t)n * 8 * sizeof(int)));
    CUDA_CHECK(cudaHostAlloc((void**)&h_ids, (size_t)n * 8 * sizeof(int), cudaHostAllocDefault));
    size_t free_now = 0, total_now = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_now, &total_now));
    log("layer-major prefill: offset %d, %d tokens, tile %d, %.1f MiB window hidden/routing, %.1f MiB CUDA free after state",
        window_start, n, tile, ((size_t)n * (3 * kE * sizeof(float) + 8 * (sizeof(float) + 2 * sizeof(int)))) / 1048576.0,
        free_now / 1048576.0);

    // Embeddings enter one persistent full-prompt hidden tensor. It is updated in place after each complete layer.
    std::vector<float> emb((size_t)tile * kE);
    for (int done = 0; done < n; done += tile) {
        const int T = std::min(tile, n - done);
        for (int t = 0; t < T; ++t) E.model_.embed(tokens[window_start + done + t], emb.data() + (size_t)t * kE);
        CUDA_CHECK(cudaMemcpyAsync(full_x + (size_t)done * kE, emb.data(), (size_t)T * kE * sizeof(float),
                                   cudaMemcpyHostToDevice, s));
        // Do not rely on pageable-memory staging behavior before reusing this source for the next tile.
        CUDA_CHECK(cudaStreamSynchronize(s));
    }

    std::vector<int> perm((size_t)n * 8), inv((size_t)n * 8);
    int profile_level = 0;
    if (const char* v = std::getenv("STRATA_PREFILL_PROFILE")) profile_level = std::atoi(v);
    std::unique_ptr<LayerMajorProfile> profile;
    if (profile_level > 0) {
        profile = std::make_unique<LayerMajorProfile>(profile_level >= 2);
        log("PREFILL_PROFILE_INFO level=%d window=%d: GPU fields are stream elapsed spans, not kernel busy; "
            "hostgap includes metadata copies/setup; hostplan includes CPU planning and API calls; "
            "copy service overlaps compute and must not be added to wall time; exposed_copy_wait is a "
            "compute-stream wait span including scheduling/event overhead; detailed samples are layers 0 and 3; "
            "waits are per group with group prefetch, otherwise per streamed expert; copy spans count H2D only; "
            "unsampled detail fields are -1; allocation, embeddings, MTP and phase restore are outside layer spans",
            profile_level, window_start);
    }
    std::unique_ptr<MixerProfile> mixer_profile;
    if (const char* v = std::getenv("STRATA_PREFILL_MIXER_PROFILE"); v && std::atoi(v) > 0) {
        mixer_profile = std::make_unique<MixerProfile>((n + tile - 1) / tile);
        log("PREFILL_MIXER_PROFILE_INFO window=%d: sampled layers 0 and 3; total covers every tile; "
            "first/last exclude once-per-layer dequant; rope_kv_prep includes Q/K norm, RoPE and KV store; "
            "all times are same-stream elapsed spans including event/scheduling overhead, not hardware busy; "
            "events read after existing end-layer synchronization; no new host synchronization", window_start);
    }
    for (int il = 0; il < c.n_layer; ++il) {
        const LayerW& L = E.model_.layers[il];
        MixerProfile* mp = mixer_profile && (il == 0 || il == 3) ? mixer_profile.get() : nullptr;
        P.mixer_profile = mp;
        const bool profile_detail = profile_level >= 2 && (il == 0 || il == 3);
        const double layer_host_start = profile ? now_ms() : 0;
        if (profile) CUDA_CHECK(cudaEventRecord(profile->phase[0], s));
        const bool diagnostic = !P.diagnostic_dir.empty() && E.n_past_ == 0 && window_start == 0 &&
                                (P.diagnostic_layer < 0 || P.diagnostic_layer == il);
        // Prepare both mixer matrices once. The ordinary chunk-major path expands them once per chunk.
        if (mp) CUDA_CHECK(cudaEventRecord(mp->dequant[0], s));
        if (P.dense_f32) dequant_q8_half_f32(L.attn ? L.qkv : L.qkvz, P.W16, s);
        else dequant_q8(L.attn ? L.qkv : L.qkvz, P.W16, s);
        if (mp) CUDA_CHECK(cudaEventRecord(mp->dequant[1], s));
        dequant_q8(L.attn ? L.wo : L.ssm_out, P.W16b, s);
        if (mp) CUDA_CHECK(cudaEventRecord(mp->dequant[2], s));
        for (int done = 0; done < n; done += tile) {
            const int T = std::min(tile, n - done);
            P.mixer_profile_tile = done / tile;
            if (mp) mp->mark(P.mixer_profile_tile, MixerProfile::D2D_BEGIN, s);
            CUDA_CHECK(cudaMemcpyAsync(P.X, full_x + (size_t)done * kE, (size_t)T * kE * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
            if (mp) mp->mark(P.mixer_profile_tile, MixerProfile::D2D_IN, s);
            mixer_prepared(il, T, E.n_past_ + window_start + done);
            CUDA_CHECK(cudaMemcpyAsync(full_x + (size_t)done * kE, P.X, (size_t)T * kE * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
            if (mp) mp->mark(P.mixer_profile_tile, MixerProfile::D2D_OUT, s);
        }
        P.mixer_profile = nullptr;
        if (diagnostic) diagnostic_dump(P.diagnostic_dir, "layer", il, "postmixer", full_x, n, kE, s);
        if (profile) CUDA_CHECK(cudaEventRecord(profile->phase[1], s));
        const double mixer_host_end = profile ? now_ms() : 0;

        // Shared-expert matrices are also expanded once. Routing metadata and normalized hidden rows are retained
        // for the complete layer, while the much larger per-route outputs stay bounded by route_cap.
        dequant_q8(L.sh_gu, P.W16, s);
        __half* w_shared_down = P.W16 + (size_t)2 * kFF * kE;
        dequant_q8(L.sh_down, w_shared_down, s);
        for (int done = 0; done < n; done += tile) {
            const int T = std::min(tile, n - done);
            CUDA_CHECK(cudaMemcpyAsync(P.X, full_x + (size_t)done * kE, (size_t)T * kE * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
            rmsnorm_rows_kernel<<<T, 256, 0, s>>>(P.X, L.post_norm, c.eps, P.Hn, P.Hh);
            gemm_f32(P.cb, L.router.w, P.Hn, P.rlog, 257, kE, T, 257);
            router_rows_kernel<<<T, 256, 0, s>>>(P.rlog, 257, P.ids, P.wts, E.route_counts_, il);
            gemm(P.cb, P.W16, P.Hh, P.shgu, 2 * kFF, kE, T, 2 * kFF, 0.0f);
            swiglu_rows_kernel<<<T, 256, 0, s>>>(P.shgu, P.act, kFF);
            gemm(P.cb, w_shared_down, P.act, P.shout, kE, kFF, T, kE, 0.0f);
            if (ordered_combine) {
                // Preserve raw inputs for the exact ordinary combine kernel. Pre-rounding sg*shared in a
                // separate kernel changes the compiler's first FMA contraction and can amplify through routing.
                CUDA_CHECK(cudaMemcpyAsync(full_acc + (size_t)done * kE, P.shout,
                                           (size_t)T * kE * sizeof(float), cudaMemcpyDeviceToDevice, s));
                CUDA_CHECK(cudaMemcpyAsync(full_rlog + (size_t)done * 257, P.rlog,
                                           (size_t)T * 257 * sizeof(float), cudaMemcpyDeviceToDevice, s));
            } else {
                shared_store_rows_kernel<<<T, 256, 0, s>>>(full_acc + (size_t)done * kE, P.shout, P.rlog, 257);
            }
            CUDA_CHECK(cudaMemcpyAsync(full_x + (size_t)done * kE, P.X, (size_t)T * kE * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(full_hn + (size_t)done * kE, P.Hn, (size_t)T * kE * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(full_w + (size_t)done * 8, P.wts, (size_t)T * 8 * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(h_ids + (size_t)done * 8, P.ids, (size_t)T * 8 * sizeof(int),
                                       cudaMemcpyDeviceToHost, s));
        }
        if (profile) CUDA_CHECK(cudaEventRecord(profile->phase[2], s));
        const double router_sync_start = profile ? now_ms() : 0;
        CUDA_CHECK(cudaStreamSynchronize(s));
        const double hostplan_start = profile ? now_ms() : 0;

        if (diagnostic)
            diagnostic_write(P.diagnostic_dir, "layer", il, "routerids", h_ids,
                             (size_t)std::min(n, 1024) * 8 * sizeof(int));
        int cnt[257] = {0};
        for (int a = 0; a < n * 8; ++a) ++cnt[h_ids[a] + 1];
        for (int e = 0; e < 256; ++e) cnt[e + 1] += cnt[e];
        int off[256];
        std::memcpy(off, cnt, sizeof(off));
        for (int a = 0; a < n * 8; ++a) {
            const int pos = off[h_ids[a]]++;
            perm[(size_t)pos] = a >> 3;
            inv[(size_t)a] = pos;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_perm, perm.data(), (size_t)n * 8 * sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d_inv, inv.data(), (size_t)n * 8 * sizeof(int), cudaMemcpyHostToDevice, s));
        const bool one_tile_exact = n <= tile && (int64_t)n * 8 <= route_cap;
        if (one_tile_exact)
            strata::prefill::mmq::quantize(full_hn, d_perm, P.Xq, (int)L.t_gu, kE, kE, (int64_t)n * 8, s);
        std::vector<int> order;
        for (int e = 0; e < 256; ++e) if (cnt[e + 1] > cnt[e]) order.push_back(e);

        std::vector<int> stage_of(order.size(), -1);
        size_t next_copy = 0;
        int ring = 0;
        auto issue_copies = [&](size_t upto) {
            for (; next_copy < order.size() && next_copy < upto; ++next_copy) {
                const int ex = order[next_copy];
                if (E.residency_host_[(size_t)il * 256 + ex] >= 0) continue;
                const int sl = ring++ % kStage;
                stage_of[next_copy] = sl;
                CUDA_CHECK(cudaStreamWaitEvent(P.cs, P.ev_free[sl], 0));
                if (profile_detail) CUDA_CHECK(cudaEventRecord(profile->copy_begin[next_copy], P.cs));
                CUDA_CHECK(cudaMemcpyAsync(P.stage + (size_t)sl * P.stage_bytes, L.expert_blob(ex),
                                           (size_t)L.blob_bytes, cudaMemcpyHostToDevice, P.cs));
                // End precedes ev_copied: consuming the dependency also guarantees this timing event is done.
                if (profile_detail) CUDA_CHECK(cudaEventRecord(profile->copy_end[next_copy], P.cs));
                CUDA_CHECK(cudaEventRecord(P.ev_copied[sl], P.cs));
            }
        };
        auto ahead_limit = [&](size_t i) {
            size_t k = i, used = 0;
            while (k < order.size() && used < (size_t)kStage) {
                if (E.residency_host_[(size_t)il * 256 + order[k]] < 0) ++used;
                ++k;
            }
            return k;
        };

        const size_t layer_gu_stride = (size_t)2 * L.gu_bytes;
        const size_t layer_down_stride = (size_t)L.down_bytes;
        float* mmq_h = reinterpret_cast<float*>(P.Xe);
        // Pinned H2D sources must remain immutable until the transfer completes. Build every group's bounds
        // before enqueueing GPU work instead of overwriting one pinned buffer while a previous copy is pending.
        struct RowPass { size_t bounds; int first, last, max_rows; };
        struct Group { size_t begin, end, pass_begin, pass_end; };
        std::vector<Group> groups;
        std::vector<RowPass> passes;
        std::vector<int> host_bounds;
        int hot_experts = 0, extra_passes = 0, max_expert_rows = 0;
        for (size_t j0 = 0; j0 < order.size();) {
            size_t j1 = j0;
            int rows = 0;
            while (j1 < order.size() && j1 - j0 < kMMQGroup) {
                const int ex = order[j1];
                const int add = cnt[ex + 1] - cnt[ex];
                if (j1 > j0 && rows + add > route_cap) break;
                rows += add;
                ++j1;
            }
            const int ngx = (int)(j1 - j0);
            const int r0 = cnt[order[j0]], r1 = cnt[order[j1 - 1] + 1], nr = r1 - r0;
            const size_t pass_begin = passes.size();
            if (nr > route_cap) {
                // An individually hot expert can exceed the routed-row workspace even though the prompt fits.
                // Gather its weights once, then reuse them for bounded activation/output slices below.
                SQ_CHECK(ngx == 1 && !one_tile_exact, "oversized routed group must contain one expert");
                ++hot_experts;
                max_expert_rows = std::max(max_expert_rows, nr);
                for (int first = r0; first < r1;) {
                    const int last = first + std::min(route_cap, r1 - first);
                    const int count = last - first;
                    const size_t offset = host_bounds.size();
                    host_bounds.insert(host_bounds.end(), {0, count, 0, count});
                    passes.push_back({offset, first, last, count});
                    first = last;
                }
                extra_passes += (int)(passes.size() - pass_begin - 1);
            } else {
                const size_t offset = host_bounds.size();
                host_bounds.resize(offset + 2 * ngx + 2);
                int max_rows = 0;
                for (int q = 0; q < ngx; ++q) {
                    const int ex = order[j0 + q];
                    host_bounds[offset + q] = one_tile_exact ? cnt[ex] : cnt[ex] - r0;
                    host_bounds[offset + ngx + 1 + q] = cnt[ex] - r0;
                    max_rows = std::max(max_rows, cnt[ex + 1] - cnt[ex]);
                }
                max_expert_rows = std::max(max_expert_rows, max_rows);
                host_bounds[offset + ngx] = one_tile_exact ? r1 : nr;
                host_bounds[offset + 2 * ngx + 1] = nr;
                passes.push_back({offset, r0, r1, max_rows});
            }
            groups.push_back({j0, j1, pass_begin, passes.size()});
            j0 = j1;
        }
        const size_t nbounds = host_bounds.size();
        if (nbounds > P.mmq_bounds_capacity) {
            // Prior-layer work is already complete, so neither allocation is still in use. A low diagnostic
            // route cap or a very long skewed prompt can require more metadata than the ordinary 1024 entries.
            CUDA_CHECK(cudaFree(P.mmq_bounds));
            CUDA_CHECK(cudaFreeHost(P.h_mmq_bounds));
            P.mmq_bounds_capacity = nbounds;
            CUDA_CHECK(cudaMalloc(&P.mmq_bounds, nbounds * sizeof(int)));
            CUDA_CHECK(cudaHostAlloc((void**)&P.h_mmq_bounds, nbounds * sizeof(int), cudaHostAllocDefault));
        }
        std::memcpy(P.h_mmq_bounds, host_bounds.data(), nbounds * sizeof(int));
        CUDA_CHECK(cudaMemcpyAsync(P.mmq_bounds, P.h_mmq_bounds, nbounds * sizeof(int),
                                   cudaMemcpyHostToDevice, s));
        if (hot_experts > 0)
            log("layer-major layer %d: %d hot experts, max %d routes, cap %d, %d extra row subpasses (weights gathered once)",
                il, hot_experts, max_expert_rows, route_cap, extra_passes);
        if (profile) CUDA_CHECK(cudaEventRecord(profile->phase[3], s));
        const double hostplan_end = profile ? now_ms() : 0;
        auto issue_group_copy = [&](size_t g) {
            if (g >= groups.size()) return;
            const int bank = (int)(g % 2);
            const Group& group = groups[g];
            uint8_t* gu_dst = bank ? P.mmq_gu_alt : P.mmq_gu;
            uint8_t* down_dst = bank ? P.mmq_down_alt : P.mmq_down;
            // The recorded free generation covers every row subpass of the previous user of this bank.
            CUDA_CHECK(cudaStreamWaitEvent(P.cs, P.mmq_free[bank], 0));
            for (size_t j = group.begin; j < group.end; ++j) {
                const int ex = order[j];
                const int slot = E.residency_host_[(size_t)il * 256 + ex];
                const uint8_t* source = slot >= 0 ? E.slot_base_[il] + (size_t)slot * L.blob_bytes : L.expert_blob(ex);
                const cudaMemcpyKind kind = slot >= 0 ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
                const size_t q = j - group.begin;
                if (slot < 0) {
                    stage_of[j] = bank; // Also identifies H2D copies for the common profiler accounting.
                    if (profile_detail) CUDA_CHECK(cudaEventRecord(profile->copy_begin[j], P.cs));
                }
                // Native blobs are already [gate | up | down]. Copy unchanged packed bytes directly into the
                // next bank instead of staging and running a GPU gather. Model/cache sources remain immutable
                // throughout prefill. Pageable model storage is safe, though its H2D API may block the host.
                CUDA_CHECK(cudaMemcpyAsync(gu_dst + q * layer_gu_stride, source, layer_gu_stride, kind, P.cs));
                CUDA_CHECK(cudaMemcpyAsync(down_dst + q * layer_down_stride, source + layer_gu_stride,
                                           layer_down_stride, kind, P.cs));
                if (slot < 0 && profile_detail) CUDA_CHECK(cudaEventRecord(profile->copy_end[j], P.cs));
            }
            // Keep the prefetch stream limited to transfers and dependency events. Tail clearing
            // is queued by the consumer after this event, outside the copy pipeline.
            CUDA_CHECK(cudaEventRecord(P.mmq_ready[bank], P.cs));
        };
        if (P.group_prefetch) {
            issue_group_copy(0);
            issue_group_copy(1);
        }
        for (size_t g = 0; g < groups.size(); ++g) {
            const Group& group = groups[g];
            const size_t j0 = group.begin, j1 = group.end;
            const int ngx = (int)(j1 - j0);
            const int bank = (int)(g % 2);
            uint8_t* group_gu = P.group_prefetch && bank ? P.mmq_gu_alt : P.mmq_gu;
            uint8_t* group_down = P.group_prefetch && bank ? P.mmq_down_alt : P.mmq_down;
            const RowPass& first_pass = passes[group.pass_begin];
            if (!one_tile_exact)
                strata::prefill::mmq::quantize(full_hn, d_perm + first_pass.first, P.Xq, (int)L.t_gu,
                                               kE, kE, first_pass.last - first_pass.first, s);
            if (P.group_prefetch) {
                if (profile_detail) CUDA_CHECK(cudaEventRecord(profile->wait_begin[g], s));
                CUDA_CHECK(cudaStreamWaitEvent(s, P.mmq_ready[bank], 0));
                if (profile_detail) CUDA_CHECK(cudaEventRecord(profile->wait_end[g], s));
                // The bank remains owned by this group through every row subpass. Clear the same
                // bounded MMQ overread tails before launching its first consumer.
                zero_mmq_tails_kernel<<<1, 256, 0, s>>>(group_gu + (size_t)ngx * layer_gu_stride,
                                                       group_down + (size_t)ngx * layer_down_stride);
            } else {
                for (size_t j = j0; j < j1; ++j) {
                    issue_copies(ahead_limit(j));
                    const int ex = order[j];
                    const int slot = E.residency_host_[(size_t)il * 256 + ex];
                    const uint8_t* blob;
                    if (slot >= 0) blob = E.slot_base_[il] + (size_t)slot * L.blob_bytes;
                    else {
                        if (profile_detail) CUDA_CHECK(cudaEventRecord(profile->wait_begin[j], s));
                        CUDA_CHECK(cudaStreamWaitEvent(s, P.ev_copied[stage_of[j]], 0));
                        if (profile_detail) CUDA_CHECK(cudaEventRecord(profile->wait_end[j], s));
                        blob = P.stage + (size_t)stage_of[j] * P.stage_bytes;
                    }
                    const size_t q = j - j0;
                    strata::prefill::mmq::gather_native(blob, blob + L.gu_bytes, (size_t)L.gu_bytes,
                        blob + 2 * L.gu_bytes, (size_t)L.down_bytes,
                        P.mmq_gu + q * layer_gu_stride, P.mmq_down + q * layer_down_stride, s);
                    if (slot < 0) CUDA_CHECK(cudaEventRecord(P.ev_free[stage_of[j]], s));
                }
                CUDA_CHECK(cudaMemsetAsync(P.mmq_gu + (size_t)ngx * layer_gu_stride, 0, kMMQTail, s));
                CUDA_CHECK(cudaMemsetAsync(P.mmq_down + (size_t)ngx * layer_down_stride, 0, kMMQTail, s));
            }
            for (size_t ip = group.pass_begin; ip < group.pass_end; ++ip) {
                const RowPass& pass = passes[ip];
                const int r0 = pass.first, r1 = pass.last, nr = r1 - r0, max_rows = pass.max_rows;
                const int* bounds = P.mmq_bounds + pass.bounds;
                // The first slice was quantized before the gather; subsequent slices reuse the same expert weights.
                if (ip != group.pass_begin)
                    strata::prefill::mmq::quantize(full_hn, d_perm + r0, P.Xq, (int)L.t_gu, kE, kE, nr, s);
                strata::prefill::mmq::Product gu{};
                gu.w = group_gu; gu.type = (int)L.t_gu; gu.w_rows = 2 * kFF; gu.w_cols = kE;
                gu.expert_bytes = layer_gu_stride; gu.n = ngx; gu.xq = P.Xq; gu.bounds = bounds;
                gu.ids = P.mmq_identity; gu.total_rows = one_tile_exact ? (int64_t)n * 8 : nr;
                gu.max_rows = max_rows; gu.dst = P.gu; gu.ld_dst = 2 * kFF;
                P.mmq->run(gu, s);
                strata::prefill::mmq::swiglu(P.gu + (one_tile_exact ? (size_t)r0 * 2 * kFF : 0),
                                             mmq_h + (one_tile_exact ? (size_t)r0 * kFF : 0), nr, kFF, false, s);
                strata::prefill::mmq::quantize(mmq_h + (one_tile_exact ? (size_t)r0 * kFF : 0), nullptr,
                                               P.Hq, (int)L.t_down, kFF, kFF, nr, s);
                strata::prefill::mmq::Product down{};
                down.w = group_down; down.type = (int)L.t_down; down.w_rows = kE; down.w_cols = kFF;
                down.expert_bytes = layer_down_stride; down.n = ngx; down.xq = P.Hq;
                down.bounds = bounds + ngx + 1;
                down.ids = P.mmq_identity; down.total_rows = nr; down.max_rows = max_rows;
                // The ordered path writes directly into retained global route rows, avoiding a group-output copy.
                down.dst = ordered_combine ? full_moe + (size_t)r0 * kE :
                                             P.moe + (one_tile_exact ? (size_t)r0 * kE : 0);
                down.ld_dst = kE;
                P.mmq->run(down, s);
                if (!ordered_combine && !one_tile_exact)
                    routed_group_add_kernel<<<n, 256, 0, s>>>(full_acc, P.moe, d_inv, full_w, n, r0, r1);
            }
            if (P.group_prefetch) {
                // Queue this generation only after the current ready wait was latched and every consumer of
                // the bank was submitted. Re-recording ready for g+2 before that wait would race event reuse.
                CUDA_CHECK(cudaEventRecord(P.mmq_free[bank], s));
                issue_group_copy(g + 2);
            }
        }
        if (ordered_combine)
            moe_combine_rows_kernel<<<n, 256, 0, s>>>(full_x, full_moe, d_inv, full_w, full_acc, full_rlog, 257);
        else if (one_tile_exact)
            moe_combine_rows_kernel<<<n, 256, 0, s>>>(full_x, P.moe, d_inv, full_w, P.shout, P.rlog, 257);
        else
            add_rows_kernel<<<n, 256, 0, s>>>(full_x, full_acc, n);
        if (diagnostic) diagnostic_dump(P.diagnostic_dir, "layer", il, "postmoe", full_x, n, kE, s);
        if (profile) CUDA_CHECK(cudaEventRecord(profile->phase[4], s));
        const double layer_sync_start = profile ? now_ms() : 0;
        CUDA_CHECK(cudaStreamSynchronize(s));
        if (profile) {
            const double layer_host_end = now_ms();
            int copies = 0;
            double copy_ms = profile_detail ? 0 : -1, wait_ms = profile_detail ? 0 : -1;
            for (size_t j = 0; j < stage_of.size(); ++j) {
                if (stage_of[j] < 0) continue;
                ++copies;
                if (profile_detail) {
                    copy_ms += LayerMajorProfile::elapsed(profile->copy_begin[j], profile->copy_end[j]);
                    if (!P.group_prefetch)
                        wait_ms += LayerMajorProfile::elapsed(profile->wait_begin[j], profile->wait_end[j]);
                }
            }
            if (P.group_prefetch && profile_detail)
                for (size_t g = 0; g < groups.size(); ++g)
                    wait_ms += LayerMajorProfile::elapsed(profile->wait_begin[g], profile->wait_end[g]);
            log("PREFILL_PROFILE layer=%d window=%d tokens=%d tile=%d attention=%d detailed=%d diagnostic=%d "
                "wall_ms=%.4f layer_stream_ms=%.4f mixer_ms=%.4f shared_ms=%.4f hostgap_stream_ms=%.4f moe_ms=%.4f "
                "mixer_enqueue_ms=%.4f shared_enqueue_ms=%.4f router_syncblock_ms=%.4f hostplan_ms=%.4f "
                "moe_enqueue_ms=%.4f syncblock_ms=%.4f exposed_copy_wait_ms=%.4f sampled_copy_ms=%.4f "
                "copies=%d bytes=%llu pinned=%d experts=%zu groups=%zu passes=%zu group_prefetch=%d",
                il, window_start, n, tile, L.attn ? 1 : 0, profile_detail ? 1 : 0, diagnostic ? 1 : 0,
                layer_host_end - layer_host_start, LayerMajorProfile::elapsed(profile->phase[0], profile->phase[4]),
                LayerMajorProfile::elapsed(profile->phase[0], profile->phase[1]),
                LayerMajorProfile::elapsed(profile->phase[1], profile->phase[2]),
                LayerMajorProfile::elapsed(profile->phase[2], profile->phase[3]),
                LayerMajorProfile::elapsed(profile->phase[3], profile->phase[4]),
                mixer_host_end - layer_host_start, router_sync_start - mixer_host_end,
                hostplan_start - router_sync_start, hostplan_end - hostplan_start,
                layer_sync_start - hostplan_end, layer_host_end - layer_sync_start, wait_ms, copy_ms,
                copies, (unsigned long long)copies * (unsigned long long)L.blob_bytes,
                L.experts_pinned ? 1 : 0, order.size(), groups.size(), passes.size(), P.group_prefetch ? 1 : 0);
        }
        if (mp) mp->report(il, window_start, n, tile, L.attn);
    }

    // The trunk is complete. MTP remains tiled because it is only one layer; the main 40-layer weight traffic has
    // already been amortized. The last normalized row is left in E.xq_ for a phase-loaned head restore.
    for (int done = 0; done < n;) {
        const int T = std::min(tile, n - done);
        const int p0 = E.n_past_ + window_start + done;
        CUDA_CHECK(cudaMemcpyAsync(P.X, full_x + (size_t)done * kE, (size_t)T * kE * sizeof(float),
                                   cudaMemcpyDeviceToDevice, s));
        if (window_start + done + T == total_n) {
            k_rmsnorm_q(P.X + (size_t)(T - 1) * kE, E.model_.output_norm, c.eps, E.xn_, E.xq_, 1, s);
            if (!E.model_.phase_lm_head) k_gemv_q8(E.model_.lm_head, E.xq_, E.logits_, 1, s);
        }
        if (E.mtp_on_) mtp(tokens, total_n, window_start + done, T, p0);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaGetLastError());
        done += T;
    }
    CUDA_CHECK(cudaFree(full_x));
    CUDA_CHECK(cudaFree(full_hn));
    CUDA_CHECK(cudaFree(full_acc));
    CUDA_CHECK(cudaFree(full_w));
    if (full_moe) CUDA_CHECK(cudaFree(full_moe));
    if (full_rlog) CUDA_CHECK(cudaFree(full_rlog));
    CUDA_CHECK(cudaFree(d_perm));
    CUDA_CHECK(cudaFree(d_inv));
    CUDA_CHECK(cudaFreeHost(h_ids));
#endif
}

void Prefill::run(const int* tokens, int n) {
    if (p_->layer_major) {
        run_layer_major(tokens, n);
        return;
    }
    // with MTP, a chunk's MTP pass also carries the pending pairs of the previous one
    const int cmax = e_.mtp_on_ ? p_->C - kMaxT : p_->C;
    for (int done = 0; done < n;) {
        const int T = std::min(cmax, n - done);
        Impl& P = *p_;
        Engine& E = e_;
        const Config& c = E.model_.cfg;
        const float eps = c.eps;
        const int p0 = E.n_past_ + done;
        cudaStream_t s = P.s;
        // ---- embeddings (host dequant -> device)
        {
            std::vector<float> emb((size_t)T * kE);
            for (int t = 0; t < T; ++t) E.model_.embed(tokens[done + t], emb.data() + (size_t)t * kE);
            CUDA_CHECK(cudaMemcpyAsync(P.X, emb.data(), emb.size() * 4, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
        for (int il = 0; il < c.n_layer; ++il) layer(il, T, p0);
        // ---- logits of the last token of the chunk (only the final chunk's matter)
        if (done + T == n) {
            k_rmsnorm_q(P.X + (size_t)(T - 1) * kE, E.model_.output_norm, eps, E.xn_, E.xq_, 1, s);
            // A phase-loaned head is restored after this workspace is destroyed. E.xq_ is a persistent decode
            // buffer, so preserving this one normalized row is sufficient to calculate final logits then.
            if (!E.model_.phase_lm_head) k_gemv_q8(E.model_.lm_head, E.xq_, E.logits_, 1, s);
        }
        if (E.mtp_on_) mtp(tokens, n, done, T, p0);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaGetLastError());
        done += T;
    }
}

}  // namespace sq
