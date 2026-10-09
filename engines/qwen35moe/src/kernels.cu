// kernels.cu - decode-step kernels (1..kMaxT tokens) for Qwen3.6-35B-A3B.  See kernels.cuh.
//
// The K-quant dot products follow the block layouts of ggml (MIT); the lane mapping is this engine's own:
// 8 lanes per 256-value super-block, 16 bytes of quants per lane, activations quantized to int8 per 32 values.
#include "kernels.cuh"
#include "actq_readonly.cuh"
#include "dense_q6.cuh"

#include "common.hpp"
#include "quant.hpp"
#include "iq_native.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstring>

namespace sq {

namespace {

constexpr int kE = 2048;       // n_embd
constexpr int kHD = 256;       // attention head dim
constexpr int kNH = 16;        // q heads
constexpr int kNKV = 2;        // kv heads
constexpr int kVH = 32;        // delta-net value heads
constexpr int kKH = 16;        // delta-net key heads
constexpr int kDS = 128;       // delta-net head dim
constexpr int kConv = 8192;    // conv channels
constexpr int kFF = 512;       // expert hidden
constexpr int kQKVZ = kConv + 4096;           // delta-net projection row (q k v | z)
constexpr int kQKV = kNH * 2 * kHD + 2 * kNKV * kHD;   // attention projection row (q+gate | k | v)
constexpr int kAttSplits = 64;  // decode attention: at most this many key ranges (blocks per kv head)
constexpr int kAttSub = 64;     // keys per online-softmax step inside a range

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

/// Sum over the whole block (blockDim.x multiple of 32, <= 1024).  `sh` needs 32 floats.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = lane < nw ? sh[lane] : 0.0f;
    v = warp_sum(v);
    return v;
}

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

/// Quantize one value per lane: the 32 lanes of a warp form one q8 block.
__device__ __forceinline__ void quant_lane(float v, ActQ out, int idx /* value index, block = idx/32 */) {
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.0f;
    const int q = amax > 0.0f ? __float2int_rn(v / d) : 0;
    const float sum = warp_sum((float)q);
    out.q[idx] = (int8_t)q;
    if ((threadIdx.x & 31) == 0) {
        out.d[idx >> 5] = d;
        out.s[idx >> 5] = d * sum;
    }
}

/// Quantize 8 values per lane: 4 lanes form one q8 block.  `idx0` = index of v[0]; lanes are consecutive.
__device__ __forceinline__ void quant8(const float v[8], ActQ out, int idx0) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float d = amax / 127.0f;
    const float id = amax > 0.0f ? 1.0f / d : 0.0f;
    int sum = 0;
    char4 packed[2];
    int8_t* pk = (int8_t*)packed;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = __float2int_rn(v[i] * id);
        pk[i] = (int8_t)q;
        sum += q;
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    *(int2*)(out.q + idx0) = *(int2*)packed;
    if ((idx0 & 31) == 0) {
        out.d[idx0 >> 5] = d;
        out.s[idx0 >> 5] = d * (float)sum;
    }
}

// =====================================================================================================================
// elementwise / norms
// =====================================================================================================================
__global__ void embed_in_kernel(const StepParams* P, float* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i < kE) x[t * kE + i] = P->emb[t][i];
}

/// n = 2048, 256 threads, 8 values each; one block per token.
__global__ void rmsnorm_q_kernel(const float* __restrict__ x, const float* __restrict__ w, float eps,
                                 float* __restrict__ out_f, ActQ out_q) {
    __shared__ float sh[32];
    const int i0 = threadIdx.x * 8, t = blockIdx.x;
    x += t * kE;
    if (out_f) out_f += t * kE;
    out_q = out_q.row(t, kE);
    float v[8];
    *(float4*)(v + 0) = *(const float4*)(x + i0);
    *(float4*)(v + 4) = *(const float4*)(x + i0 + 4);
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / kE + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = v[i] * r * w[i0 + i];
    if (out_f) {
        *(float4*)(out_f + i0) = *(float4*)(v + 0);
        *(float4*)(out_f + i0 + 4) = *(float4*)(v + 4);
    }
    quant8(v, out_q, i0);
}

__global__ void add_rmsnorm_q_kernel(float* __restrict__ x, const float* __restrict__ a, const float* __restrict__ w,
                                     float eps, float* __restrict__ xn, ActQ xq) {
    __shared__ float sh[32];
    const int i0 = threadIdx.x * 8, t = blockIdx.x;
    x += t * kE;
    a += t * kE;
    xn += t * kE;
    xq = xq.row(t, kE);
    float v[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = x[i0 + i] + a[i0 + i];
        x[i0 + i] = v[i];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / kE + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = v[i] * r * w[i0 + i];
        xn[i0 + i] = v[i];
    }
    quant8(v, xq, i0);
}

/// MTP input: grid (T, 2).  Half 0: rmsnorm(embedding) * enorm, half 1: rmsnorm(h) * hnorm; quantized side by side
/// as one 4096-vector per token (the order of the MTP's eh_proj input).
__global__ void mtp_in_kernel(const StepParams* P, const float* __restrict__ h, const float* __restrict__ enorm,
                              const float* __restrict__ hnorm, float eps, ActQ out) {
    __shared__ float sh[32];
    const int t = blockIdx.x, half = blockIdx.y, i0 = threadIdx.x * 8;
    const float* x = half ? h + t * kE : P->emb[t];
    const float* w = half ? hnorm : enorm;
    float v[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = x[i0 + i];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / kE + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = v[i] * r * w[i0 + i];
    quant8(v, out, t * 2 * kE + half * kE + i0);
}

// =====================================================================================================================
// GEMV: Q8_0 (SoA) x ActQ, and F32
// =====================================================================================================================
constexpr int kGemvWarps = 8;

/// One row per warp; the NT activation vectors sit in shared memory and every weight load serves all of them.
template <int NT, bool ReadOnly = false>
__global__ void __launch_bounds__(kGemvWarps * 32) gemv_q8_kernel(const int8_t* __restrict__ wq, const __half* __restrict__ wd,
                                                                  ActQ a, float* __restrict__ y, int rows, int cols) {
    extern __shared__ int4 smem[];
    const int nvec = cols >> 4;     // 16-byte chunks per row
    const int nsc = cols >> 5;      // scales per row
    int4* sa = smem;
    float* sd = (float*)(smem + NT * nvec);
    for (int i = threadIdx.x; i < NT * nvec; i += blockDim.x) sa[i] = actq_load<ReadOnly>((const int4*)a.q + i);
    for (int i = threadIdx.x; i < NT * nsc; i += blockDim.x) sd[i] = actq_load<ReadOnly>(a.d + i);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * kGemvWarps + warp;
    if (row >= rows) return;
    const int4* wr = (const int4*)(wq + (size_t)row * cols);
    const __half* wdr = wd + (size_t)row * nsc;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
#pragma unroll 4
    for (int i = lane; i < nvec; i += 32) {
        const int4 w4 = __ldg(wr + i);
        const float ws = __half2float(wdr[i >> 1]);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int4 a4 = sa[t * nvec + i];
            int s = __dp4a(w4.x, a4.x, 0);
            s = __dp4a(w4.y, a4.y, s);
            s = __dp4a(w4.z, a4.z, s);
            s = __dp4a(w4.w, a4.w, s);
            acc[t] += (float)s * ws * sd[t * nsc + (i >> 1)];
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float r = warp_sum(acc[t]);
        if (lane == 0) y[(size_t)t * rows + row] = r;
    }
}

/// Small fp32 matrices (router, beta/alpha): 2 rows per 256-thread block, 4 warps per row splitting the columns,
/// so that even 64 rows spread over enough warps to hide the load latency.
template <int NT>
__global__ void __launch_bounds__(256) gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                       float* __restrict__ y, int rows, int cols) {
    __shared__ float part[NT][8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 2 + (warp >> 2), q = warp & 3;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    if (row < rows) {
        const int n4 = cols >> 2, per = n4 >> 2;   // float4s per row / per warp (cols % 16 == 0)
        const float4* wr = (const float4*)(W + (size_t)row * cols) + q * per;
#pragma unroll 4
        for (int i = lane; i < per; i += 32) {
            const float4 w = __ldg(wr + i);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 v = __ldg((const float4*)(x + (size_t)t * cols) + q * per + i);
                acc[t] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float r = warp_sum(acc[t]);
        if (lane == 0) part[t][warp] = r;
    }
    __syncthreads();
    if (threadIdx.x < 2 * NT) {
        const int t = threadIdx.x >> 1, h = threadIdx.x & 1;
        const int r = blockIdx.x * 2 + h;
        if (r < rows) {
            const float* pp = part[t] + h * 4;
            y[(size_t)t * rows + r] = (pp[0] + pp[1]) + (pp[2] + pp[3]);
        }
    }
}

// =====================================================================================================================
// K-quant dot products, 8 lanes per super-block (c = lane within the 8)
// =====================================================================================================================
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

/// Activation pointers for one super-block: 256 int8, 8 scales, 8 sums.
struct ActSB {
    const int8_t* q;
    const float* d;
    const float* s;
};

__device__ __forceinline__ float dot_q4k_c(const uint8_t* blk, int c, ActSB a) {
    const float2 dm = __half22float2(*(const __half2*)blk);
    const int j = c >> 1, p0 = (c & 1) * 16;
    int sc0, m0, sc1, m1;
    scale_min_k4(2 * j, blk + 4, sc0, m0);
    scale_min_k4(2 * j + 1, blk + 4, sc1, m1);
    const int4 q = *(const int4*)(blk + 16 + 16 * c);
    const int4 alo = *(const int4*)(a.q + 64 * j + p0);
    const int4 ahi = *(const int4*)(a.q + 64 * j + 32 + p0);
    int sl = __dp4a(q.x & 0x0F0F0F0F, alo.x, 0), sh = __dp4a((q.x >> 4) & 0x0F0F0F0F, ahi.x, 0);
    sl = __dp4a(q.y & 0x0F0F0F0F, alo.y, sl); sh = __dp4a((q.y >> 4) & 0x0F0F0F0F, ahi.y, sh);
    sl = __dp4a(q.z & 0x0F0F0F0F, alo.z, sl); sh = __dp4a((q.z >> 4) & 0x0F0F0F0F, ahi.z, sh);
    sl = __dp4a(q.w & 0x0F0F0F0F, alo.w, sl); sh = __dp4a((q.w >> 4) & 0x0F0F0F0F, ahi.w, sh);
    float r = dm.x * ((float)(sc0 * sl) * a.d[2 * j] + (float)(sc1 * sh) * a.d[2 * j + 1]);
    if (p0 == 0) r -= dm.y * ((float)m0 * a.s[2 * j] + (float)m1 * a.s[2 * j + 1]);
    return r;
}

__device__ __forceinline__ float dot_q5k_c(const uint8_t* blk, int c, ActSB a) {
    const float2 dm = __half22float2(*(const __half2*)blk);
    const int j = c >> 1, p0 = (c & 1) * 16;
    int sc0, m0, sc1, m1;
    scale_min_k4(2 * j, blk + 4, sc0, m0);
    scale_min_k4(2 * j + 1, blk + 4, sc1, m1);
    const int4 q = *(const int4*)(blk + 48 + 16 * c);
    const int4 qh = *(const int4*)(blk + 16 + p0);
    const int4 alo = *(const int4*)(a.q + 64 * j + p0);
    const int4 ahi = *(const int4*)(a.q + 64 * j + 32 + p0);
    const int b0 = 2 * j, b1 = 2 * j + 1;
#define Q5L(qq, hh) (((qq) & 0x0F0F0F0F) | ((((hh) >> b0) & 0x01010101) << 4))
#define Q5H(qq, hh) ((((qq) >> 4) & 0x0F0F0F0F) | ((((hh) >> b1) & 0x01010101) << 4))
    int sl = __dp4a(Q5L(q.x, qh.x), alo.x, 0), sh = __dp4a(Q5H(q.x, qh.x), ahi.x, 0);
    sl = __dp4a(Q5L(q.y, qh.y), alo.y, sl); sh = __dp4a(Q5H(q.y, qh.y), ahi.y, sh);
    sl = __dp4a(Q5L(q.z, qh.z), alo.z, sl); sh = __dp4a(Q5H(q.z, qh.z), ahi.z, sh);
    sl = __dp4a(Q5L(q.w, qh.w), alo.w, sl); sh = __dp4a(Q5H(q.w, qh.w), ahi.w, sh);
#undef Q5L
#undef Q5H
    float r = dm.x * ((float)(sc0 * sl) * a.d[2 * j] + (float)(sc1 * sh) * a.d[2 * j + 1]);
    if (p0 == 0) r -= dm.y * ((float)m0 * a.s[2 * j] + (float)m1 * a.s[2 * j + 1]);
    return r;
}

__device__ __forceinline__ int ld_int_b2(const uint8_t* p) {
    const uint16_t* h = (const uint16_t*)p;
    return (int)h[0] | ((int)h[1] << 16);
}

__device__ __forceinline__ float dot_q6k_c(const uint8_t* blk, int c, ActSB a) {
    // lane c covers values 128h + 32qd + [0, 32)
    const int h = c >> 2, qd = c & 3;
    const uint8_t* ql = blk + 64 * h + 32 * (qd & 1);
    const uint8_t* qh = blk + 128 + 32 * h;
    const int nsh = 4 * (qd >> 1), hsh = 2 * qd;
    const int8_t* sc = (const int8_t*)(blk + 192) + 8 * h + 2 * qd;
    const float d = __half2float(*(const __half*)(blk + 208));
    const int4* ap = (const int4*)(a.q + 128 * h + 32 * qd);
    const int4 a0 = ap[0], a1 = ap[1];
    const int av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int vl = (ld_int_b2(ql + 4 * k) >> nsh) & 0x0F0F0F0F;
        const int vh = ((ld_int_b2(qh + 4 * k) >> hsh) & 0x03030303) << 4;
        const int v = __vsubss4(vl | vh, 0x20202020);
        if (k < 4) s0 = __dp4a(v, av[k], s0);
        else s1 = __dp4a(v, av[k], s1);
    }
    return d * a.d[4 * h + qd] * (float)(sc[0] * s0 + sc[1] * s1);
}

template <uint32_t T>
__device__ __forceinline__ float dot_c(const uint8_t* blk, int c, ActSB a) {
    if constexpr (T == T_Q4_K) return dot_q4k_c(blk, c, a);
    else if constexpr (T == T_Q5_K) return dot_q5k_c(blk, c, a);
    else if constexpr (T == T_IQ2_S || T == T_IQ3_S || T == T_IQ4_XS || T == T_Q2_K || T == T_Q3_K)
        return iq_native::dot_32<T>(blk, c, a.q + 32 * c, a.d[c]);
    else return dot_q6k_c(blk, c, a);
}

template <uint32_t T> constexpr int blk_bytes() {
    return T == T_Q2_K ? 84 : T == T_Q3_K ? 110 :
           T == T_IQ2_S ? 82 : T == T_IQ3_S ? 110 : T == T_IQ4_XS ? 136 :
           T == T_Q4_K ? 144 : T == T_Q5_K ? 176 : 210;
}

/// Dot of one quantized row of NSB super-blocks with the activation in shared memory.  The row is split over
/// LPR = min(32, NSB * 8) lanes; returns the full sum on every one of those lanes.
template <uint32_t T, int NSB, bool Transpose = false>
__device__ __forceinline__ float row_dot(const uint8_t* row, const int8_t* aq, const float* ad, const float* as, int lane_in_row) {
    constexpr int LPR = NSB * 8 < 32 ? NSB * 8 : 32;
    constexpr int PASSES = NSB * 8 / LPR;
    float acc = 0.0f;
#pragma unroll
    for (int p = 0; p < PASSES; ++p) {
        const int sb = p * (LPR / 8) + (lane_in_row >> 3);
        const int c = lane_in_row & 7;
        if constexpr (Transpose)
            acc += iq_native::dot_32<T, NSB * 8 + 1>(row + sb * blk_bytes<T>(), c,
                                                      aq + 4 * (sb * 8 + c), ad[sb * 8 + c]);
        else acc += dot_c<T>(row + sb * blk_bytes<T>(), c, ActSB{aq + sb * 256, ad + sb * 8, as + sb * 8});
    }
#pragma unroll
    for (int o = LPR / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

// Native packed experts: unpack each lane's weight group once for every token routed to this expert.
// Keep row_dot's super-block passes and XOR reduction sequence separately for each token.
template <uint32_t Type, int NSB, int NT, bool Transpose = false>
__device__ __forceinline__ void row_dot_reuse(const uint8_t* row, const int8_t* aq, const float* ad,
                                              const int (&input_rows)[NT], int lane_in_row, float (&acc)[NT]) {
    constexpr int LPR = NSB * 8 < 32 ? NSB * 8 : 32;
    constexpr int PASSES = NSB * 8 / LPR;
    unsigned active = 0;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        acc[t] = 0.0f;
        if (input_rows[t] >= 0) active |= 1u << t;
    }
#pragma unroll
    for (int p = 0; p < PASSES; ++p) {
        const int sb = p * (LPR / 8) + (lane_in_row >> 3), group = lane_in_row & 7;
        const int8_t* token_q[NT];
        float token_d[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if constexpr (Transpose)
                token_q[t] = input_rows[t] >= 0 ? aq + input_rows[t] * ((NSB * 8 + 1) * 32) + 4 * (sb * 8 + group) : nullptr;
            else token_q[t] = input_rows[t] >= 0 ? aq + input_rows[t] * (NSB * 256) + sb * 256 + group * 32 : nullptr;
            token_d[t] = input_rows[t] >= 0 ? ad[input_rows[t] * (NSB * 8) + sb * 8 + group] : 0.0f;
        }
        // Accumulate in the same expression as row_dot's inlined scalar helper. Materializing an array
        // of rounded contributions first prevents its final multiply/add contraction on the second pass.
        iq_native::dot_32_multi<Type, NT, true, Transpose ? NSB * 8 + 1 : 1>(
            row + sb * blk_bytes<Type>(), group, token_q, token_d, active, acc);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int o = LPR / 2; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
    }
}

// =====================================================================================================================
// MoE: resident experts on the GPU
// =====================================================================================================================
// Transpose 8 int32 words/group into word planes. Consecutive dot-product lanes then read consecutive banks,
// replacing the contiguous layout's stride-eight shared loads. One padding group keeps different planes from
// all landing on the same banks during the one-time staging copy, with only 32 extra bytes per activation row.
template <int Cols>
__device__ __forceinline__ void stage_native_transpose(int8_t* shared, const int8_t* global, int rows) {
    constexpr int Groups = Cols / 32, Plane = Groups + 1;
    for (int i = threadIdx.x; i < rows * Cols / 4; i += blockDim.x) {
        const int row = i / (Cols / 4), q = i % (Cols / 4);
        ((int*)shared)[row * (8 * Plane) + (q & 7) * Plane + (q >> 3)] = ((const int*)global)[i];
    }
}

/// gate|up (1024 rows x 2048) of each hit expert: grid (1024/8, distinct experts), one row per warp.  The row is
/// dotted with every token that picked the expert (the repeated loads come from L1).
template <uint32_t T, int NT, bool Reuse = false, bool Transpose = false>
__global__ void __launch_bounds__(256) moe_gu_kernel(ActQ x, const HitList* __restrict__ hits, float* __restrict__ h_gu) {
    const int j = blockIdx.y;
    if (j >= hits->n) return;
    constexpr bool NeedSum = !Transpose; // preserve baseline layout; native transposed helpers never read ActQ.s
    constexpr int QBytes = Transpose ? kE + 32 : kE;
    __shared__ __align__(16) int8_t sq[NT * QBytes];
    __shared__ float sd[NT * kE / 32], ss[NeedSum ? NT * kE / 32 : 1];
    if constexpr (Transpose) stage_native_transpose<kE>(sq, x.q, NT);
    else for (int i = threadIdx.x; i < NT * kE / 16; i += blockDim.x) ((int4*)sq)[i] = ((const int4*)x.q)[i];
    for (int i = threadIdx.x; i < NT * kE / 32; i += blockDim.x) {
        sd[i] = x.d[i];
        if constexpr (NeedSum) ss[i] = x.s[i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    constexpr int64_t rb = (int64_t)(kE / 256) * blk_bytes<T>();
    const uint8_t* w = hits->ptr[j] + row * rb;
    if constexpr (Reuse && NT > 1) {
        int inputs[NT], pairs[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            pairs[t] = hits->pair[j][t];
            inputs[t] = pairs[t] >= 0 ? t : -1;
        }
        float values[NT];
        row_dot_reuse<T, kE / 256, NT, Transpose>(w, sq, sd, inputs, lane, values);
#pragma unroll
        for (int t = 0; t < NT; ++t)
            if (lane == 0 && pairs[t] >= 0) h_gu[pairs[t] * 2 * kFF + row] = values[t];
    } else {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int p = NT == 1 ? j : hits->pair[j][t];
            if (p < 0) continue;
            const float r = row_dot<T, kE / 256, Transpose>(w, sq + t * QBytes, sd + t * kE / 32,
                                                            ss + (NeedSum ? t * kE / 32 : 0), lane);
            if (lane == 0) h_gu[p * 2 * kFF + row] = r;
        }
    }
}

/// One block per (expert, token) pair.
__global__ void moe_act_kernel(const float* __restrict__ h_gu, const HitList* __restrict__ hits, ActQ hq) {
    const int p = blockIdx.x;
    if (p >= hits->n_pair) return;
    const int t = threadIdx.x;   // 512
    const float h = silu(h_gu[p * 2 * kFF + t]) * h_gu[p * 2 * kFF + kFF + t];
    quant_lane(h, hq, p * kFF + t);
}

/// down (2048 rows x 512) summed over the hit experts, per token: two rows per warp (16 lanes per row).
template <uint32_t T, int NT, bool Reuse = false, bool Transpose = false>
__global__ void __launch_bounds__(256) moe_down_kernel(int64_t down_off, ActQ hq, const HitList* __restrict__ hits,
                                                       float* __restrict__ out) {
    const int n = hits->n, np = hits->n_pair;
    constexpr bool NeedSum = !Transpose;
    constexpr int QBytes = Transpose ? kFF + 32 : kFF;
    __shared__ __align__(16) int8_t sq[NT * kMaxK * QBytes];
    __shared__ float sd[NT * kMaxK * kFF / 32], ss[NeedSum ? NT * kMaxK * kFF / 32 : 1];
    if constexpr (Transpose) stage_native_transpose<kFF>(sq, hq.q, np);
    else for (int i = threadIdx.x; i < np * kFF / 16; i += blockDim.x) ((int4*)sq)[i] = ((const int4*)hq.q)[i];
    for (int i = threadIdx.x; i < np * kFF / 32; i += blockDim.x) {
        sd[i] = hq.d[i];
        if constexpr (NeedSum) ss[i] = hq.s[i];
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 16 + warp * 2 + (lane >> 4);
    constexpr int64_t rb = (int64_t)(kFF / 256) * blk_bytes<T>();
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    for (int j = 0; j < n; ++j) {
        const uint8_t* w = hits->ptr[j] + down_off + row * rb;
        if constexpr (Reuse && NT > 1) {
            int inputs[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) inputs[t] = hits->pair[j][t];
            float values[NT];
            row_dot_reuse<T, kFF / 256, NT, Transpose>(w, sq, sd, inputs, lane & 15, values);
#pragma unroll
            for (int t = 0; t < NT; ++t)
                if (inputs[t] >= 0) acc[t] += hits->w[j][t] * values[t];
        } else {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int p = NT == 1 ? j : hits->pair[j][t];
                if (p < 0) continue;
                acc[t] += hits->w[j][t] * row_dot<T, kFF / 256, Transpose>(w, sq + p * QBytes,
                    sd + p * kFF / 32, ss + (NeedSum ? p * kFF / 32 : 0), lane & 15);
            }
        }
    }
    if ((lane & 15) == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) out[t * kE + row] = acc[t];
    }
}

__global__ void add_expert_output_kernel(float* __restrict__ destination, const float* __restrict__ source, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) destination[i] += source[i];
}

/// grid (n_ff / 256, T)
__global__ void swiglu_q_kernel(const float* __restrict__ gu, ActQ out, int n_ff) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i < n_ff) quant_lane(silu(gu[t * 2 * n_ff + i]) * gu[t * 2 * n_ff + n_ff + i], out, t * n_ff + i);
}

// =====================================================================================================================
// router: softmax over 256 experts, top-8, split into resident (GPU) and missing (CPU mailbox)
// =====================================================================================================================
template <bool SystemFence>
__global__ void __launch_bounds__(256) router_kernel(const float* __restrict__ logits, int layer, const int32_t* __restrict__ residency,
                                                     const CacheLayerInfo* __restrict__ cinfo, const float* __restrict__ xn,
                                                     HitList* hits, Mailbox* mb, const StepParams* P, int32_t* d_nmiss,
                                                     float* d_sg, uint32_t* counts, unsigned long long* ecount) {
    // Warp 0 selects the top 8 by logit with register-only warp argmaxes (expert id = lane + 32 i; ties -> lower id).
    // The renormalized top-8 softmax weights are exp(l_k - l_max) / sum: the full softmax denominator cancels.
    __shared__ int s_nm;
    const int t = threadIdx.x;
    if (t < 32) {
        const int lane = t;
        const float sg_logit = logits[256];
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = logits[lane + 32 * i];
        float my_l = -INFINITY, l0 = 0.0f;   // lane k keeps the k-th pick
        int my_id = 0;
#pragma unroll
        for (int k = 0; k < kMaxK; ++k) {
            float bv = v[0];
            int bi = 0;
#pragma unroll
            for (int i = 1; i < 8; ++i)
                if (v[i] > bv) { bv = v[i]; bi = i; }
            int id = lane + 32 * bi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, id, o);
                if (ov > bv || (ov == bv && oi < id)) { bv = ov; id = oi; }
            }
            if (k == 0) l0 = bv;
            if (lane == k) { my_l = bv; my_id = id; }
            if ((id & 31) == lane) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    if (i == (id >> 5)) v[i] = -INFINITY;
            }
        }
        // lanes 0..7 finish their pick in parallel (one residency load each, no serial chain)
        const bool mine = lane < kMaxK;
        const float ew = mine ? __expf(my_l - l0) : 0.0f;
        const float w = ew / warp_sum(ew);
        const int slot = mine ? residency[layer * 256 + my_id] : -1;
        // list the experts in id order, as the multi-token router does: the same summation order makes a
        // verification step bit-identical to decoding its tokens one by one
        unsigned below = 0;
#pragma unroll
        for (int k = 0; k < kMaxK; ++k)
            if (__shfl_sync(0xffffffffu, my_id, k) < my_id) below |= 1u << k;
        const unsigned hit_mask = __ballot_sync(0xffffffffu, mine && slot >= 0);
        const unsigned miss_mask = __ballot_sync(0xffffffffu, mine && slot < 0);
        if (mine) {
            if (slot >= 0) {
                const int j = __popc(hit_mask & below);
                hits->ptr[j] = cinfo[layer].base + (long long)slot * cinfo[layer].blob;
                hits->w[j][0] = w;
            } else {
                const int j = __popc(miss_mask & below);
                mb->ids[j] = my_id;
                mb->w[j][0] = w;
            }
            counts[layer * 256 + my_id] += 1;   // distinct ids: no atomics needed
        }
        if (lane == 0) {
            const int nm = __popc(miss_mask);
            hits->n = __popc(hit_mask);
            hits->n_pair = hits->n;
            mb->n_miss = nm;
            mb->n_tok = 1;
            ecount[0] += hits->n;
            ecount[1] += nm;
            *d_nmiss = nm;
            *d_sg = sigmoidf_(sg_logit);
            s_nm = nm;
        }
    }
    __syncthreads();
    if (s_nm > 0) {   // the CPU reads x only when it has experts to run
        for (int i = t; i < kE; i += blockDim.x) mb->x[0][i] = xn[i];
        if constexpr (SystemFence) __threadfence_system();   // mapped-host polling needs each writer's visibility
    }
    __syncthreads();
    if (t == 0) {
        if constexpr (SystemFence) __threadfence_system();
        *(volatile uint32_t*)&mb->ready_seq = P->seq;
    }
}

/// Router for a step of T > 1 tokens: warp t picks token t's top 8 (as above), then thread e collects expert e over
/// the tokens.  Distinct experts are listed in id order, resident ones in the HitList (with their (expert, token)
/// pair indices), missing ones in the mailbox.
template <bool SystemFence>
__global__ void __launch_bounds__(256) router_multi_kernel(const float* __restrict__ logits, int layer,
                                                           const int32_t* __restrict__ residency,
                                                           const CacheLayerInfo* __restrict__ cinfo, const float* __restrict__ xn,
                                                           HitList* hits, Mailbox* mb, const StepParams* P, int32_t* d_nmiss,
                                                           float* d_sg, uint32_t* counts, unsigned long long* ecount, int T) {
    __shared__ int s_id[kMaxT][kMaxK];
    __shared__ float s_w[kMaxT][kMaxK];
    __shared__ int s_wh[8], s_wm[8], s_wp[8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (warp < T) {
        const float* lg = logits + warp * 257;
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = lg[lane + 32 * i];
        float my_l = -INFINITY, l0 = 0.0f;
        int my_id = 0;
#pragma unroll
        for (int k = 0; k < kMaxK; ++k) {
            float bv = v[0];
            int bi = 0;
#pragma unroll
            for (int i = 1; i < 8; ++i)
                if (v[i] > bv) { bv = v[i]; bi = i; }
            int id = lane + 32 * bi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, id, o);
                if (ov > bv || (ov == bv && oi < id)) { bv = ov; id = oi; }
            }
            if (k == 0) l0 = bv;
            if (lane == k) { my_l = bv; my_id = id; }
            if ((id & 31) == lane) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    if (i == (id >> 5)) v[i] = -INFINITY;
            }
        }
        const bool mine = lane < kMaxK;
        const float ew = mine ? __expf(my_l - l0) : 0.0f;
        const float w = ew / warp_sum(ew);
        if (mine) {
            s_id[warp][lane] = my_id;
            s_w[warp][lane] = w;
        }
        if (lane == 0) d_sg[warp] = sigmoidf_(lg[256]);
    }
    __syncthreads();
    const int e = tid;
    float we[kMaxT];
    bool rt[kMaxT];
    int ntr = 0;
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
        we[t] = 0.0f;
        rt[t] = false;
        if (t < T) {
#pragma unroll
            for (int k = 0; k < kMaxK; ++k)
                if (s_id[t][k] == e) { we[t] = s_w[t][k]; rt[t] = true; }
        }
        ntr += rt[t];
    }
    const bool routed = ntr > 0;
    const int slot = routed ? residency[layer * 256 + e] : -1;
    const bool is_hit = routed && slot >= 0, is_miss = routed && slot < 0;
    const unsigned hm = __ballot_sync(0xffffffffu, is_hit), mm = __ballot_sync(0xffffffffu, is_miss);
    // pair indices: inclusive warp scan of the hit experts' token counts, then warp offsets
    int pc = is_hit ? ntr : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, pc, o);
        if (lane >= o) pc += u;
    }
    if (lane == 31) s_wp[warp] = pc;
    if (lane == 0) { s_wh[warp] = __popc(hm); s_wm[warp] = __popc(mm); }
    __syncthreads();
    int hb = 0, mbase = 0, pb = 0, nh = 0, nm = 0, npair = 0;
#pragma unroll
    for (int w8 = 0; w8 < 8; ++w8) {
        if (w8 < warp) { hb += s_wh[w8]; mbase += s_wm[w8]; pb += s_wp[w8]; }
        nh += s_wh[w8];
        nm += s_wm[w8];
        npair += s_wp[w8];
    }
    const unsigned below = (1u << lane) - 1u;
    if (is_hit) {
        const int j = hb + __popc(hm & below);
        int p = pb + pc - ntr;   // exclusive prefix
        hits->ptr[j] = cinfo[layer].base + (long long)slot * cinfo[layer].blob;
#pragma unroll
        for (int t = 0; t < kMaxT; ++t) {
            hits->w[j][t] = we[t];
            hits->pair[j][t] = rt[t] ? (int8_t)p++ : (int8_t)-1;
        }
    } else if (is_miss) {
        const int j = mbase + __popc(mm & below);
        mb->ids[j] = e;
#pragma unroll
        for (int t = 0; t < kMaxT; ++t) mb->w[j][t] = rt[t] ? we[t] : 0.0f;
    }
    if (routed) counts[layer * 256 + e] += ntr;   // one thread per expert: no atomics needed
    if (tid == 0) {
        hits->n = nh;
        hits->n_pair = npair;
        mb->n_miss = nm;
        mb->n_tok = T;
        *d_nmiss = nm;
        // User-facing hit rate counts routed (expert, token) pairs, as T==1 does.
        // nh/nm still count distinct experts for batched CPU/GPU execution, but
        // a hot expert reused by four tokens must contribute four cache hits.
        ecount[0] += npair;
        ecount[1] += T * kMaxK - npair;
    }
    if (nm > 0) {   // the CPU reads x only when it has experts to run
        for (int i = tid; i < T * kE; i += blockDim.x) mb->x[i / kE][i % kE] = xn[i];
        if constexpr (SystemFence) __threadfence_system();   // mapped-host polling needs each writer's visibility
    }
    __syncthreads();
    if (tid == 0) {
        if constexpr (SystemFence) __threadfence_system();
        *(volatile uint32_t*)&mb->ready_seq = P->seq;
    }
}

/// x += moe_gpu + moe_cpu + sigmoid(sg) * shared; then the next layer's norm.  One block of 256 threads.
__global__ void __launch_bounds__(256) combine_kernel(float* __restrict__ x, const float* __restrict__ moe_gpu,
                                                      const float* __restrict__ sh_out, const float* d_sg,
                                                      const int32_t* d_nmiss, const Result* res, const StepParams* P,
                                                      const float* __restrict__ next_w, float eps, float* __restrict__ xn,
                                                      ActQ xq, int32_t* err, unsigned long long* wait_ns) {
    __shared__ float sh[32];
    __shared__ int waited_ok;
    const int nmiss = *d_nmiss;
    const int tk = blockIdx.x;
    x += tk * kE;
    moe_gpu += tk * kE;
    sh_out += tk * kE;
    xn += tk * kE;
    xq = xq.row(tk, kE);
    if (threadIdx.x == 0) {
        waited_ok = 1;
        if (nmiss > 0) {
            const uint32_t seq = P->seq;
            const long long t0 = clock64();
            unsigned long long g0, g1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
            while (*(volatile const uint32_t*)&res->done_seq != seq) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
                if (clock64() - t0 > (1ll << 35)) {   // ~10 s: the CPU side is gone; do not hang the GPU (TDR)
                    *err = 1;
                    waited_ok = 0;
                    break;
                }
            }
            __threadfence_system();
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
            if (tk == 0) *wait_ns += g1 - g0;
        }
    }
    __syncthreads();
    const bool use_cpu = nmiss > 0 && waited_ok;
    const float sg = d_sg[tk];
    const int i0 = threadIdx.x * 8;
    float v[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float c = moe_gpu[i0 + i] + sg * sh_out[i0 + i];
        if (use_cpu) c += *(volatile const float*)&res->out[tk][i0 + i];
        v[i] = x[i0 + i] + c;
        x[i0 + i] = v[i];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / kE + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = v[i] * r * next_w[i0 + i];
        xn[i0 + i] = v[i];
    }
    quant8(v, xq, i0);
}

// =====================================================================================================================
// gated delta net
// =====================================================================================================================
/// One thread per channel; the T tokens in order.
__global__ void gdn_conv_kernel(const float* __restrict__ qkvz, float* __restrict__ cs, const float* __restrict__ cw,
                                float* __restrict__ out, int T, float* __restrict__ snap, int64_t snap_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kConv) return;
    float* st = cs + c * 3;
    const float4 w = *(const float4*)(cw + c * 4);
    float s0 = st[0], s1 = st[1], s2 = st[2];
    for (int t = 0; t < T; ++t) {
        const float in = qkvz[t * kQKVZ + c];
        const float y = w.x * s0 + w.y * s1 + w.z * s2 + w.w * in;
        s0 = s1;
        s1 = s2;
        s2 = in;
        out[t * kConv + c] = silu(y);
        if (snap && t + 1 < T) {
            float* sn = snap + t * snap_stride + c * 3;
            sn[0] = s0;
            sn[1] = s1;
            sn[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

/// grid (32 value heads, 32 column groups), 4 warps = 4 state columns per block.  State layout S[h][col][row].
/// The column stays in registers while the T tokens are applied in order.
template <int NT>
__global__ void __launch_bounds__(128) gdn_recur_kernel(const float* __restrict__ conv, const float* __restrict__ ba,
                                                        const float* __restrict__ dt_bias, const float* __restrict__ ssm_a,
                                                        float* __restrict__ state, float* __restrict__ o,
                                                        float* __restrict__ snap, int64_t snap_stride) {
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = blockIdx.y * 4 + warp;
    const int kh = h % kKH;
    float* S = state + ((size_t)h * kDS + col) * kDS;
    float s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = S[lane + 32 * i];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float* ct = conv + t * kConv;
        const float* q = ct + kh * kDS;
        const float* k = ct + kKH * kDS + kh * kDS;
        const float v = ct[2 * kKH * kDS + h * kDS + col];
        float qr[4], kr[4];
        float qs = 0.0f, ks = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            qr[i] = q[lane + 32 * i];
            kr[i] = k[lane + 32 * i];
            qs += qr[i] * qr[i];
            ks += kr[i] * kr[i];
        }
        qs = warp_sum(qs);
        ks = warp_sum(ks);
        const float qn = rsqrtf(qs + 1e-6f) * rsqrtf((float)kDS);   // l2 norm, then the 1/sqrt(d) output scale
        const float kn = rsqrtf(ks + 1e-6f);
        const float beta = sigmoidf_(ba[t * 64 + h]);
        const float a = ba[t * 64 + kVH + h] + dt_bias[h];
        const float sp = a > 20.0f ? a : log1pf(__expf(a));
        const float decay = __expf(sp * ssm_a[h]);
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kr[i] *= kn;
            kv += s[i] * kr[i];
        }
        kv = warp_sum(kv);
        const float delta = (v - decay * kv) * beta;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] = decay * s[i] + kr[i] * delta;
            out += s[i] * qr[i];
        }
        out = warp_sum(out) * qn;
        if (lane == 0) o[t * 4096 + h * kDS + col] = out;
        if (snap && t + 1 < NT) {
            float* sn = snap + t * snap_stride + ((size_t)h * kDS + col) * kDS;
#pragma unroll
            for (int i = 0; i < 4; ++i) sn[lane + 32 * i] = s[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) S[lane + 32 * i] = s[i];
}


// The four factors are invariant across all 128 state columns of one value head.
// Keep the exact original lane assignment, sum order and scalar expressions. Store
// only finalized FP32 values; recurrence still applies kn/qn in the original places.
// Caller owns T*32 float4 scratch on this stream; it must not alias input/state/output.
__global__ void gdn_prepare_kernel(const float* __restrict__ conv, const float* __restrict__ ba,
                                   const float* __restrict__ dt_bias, const float* __restrict__ ssm_a,
                                   float4* __restrict__ factors) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int kh = h % kKH;
    const float* ct = conv + t * kConv;
    const float* q = ct + kh * kDS;
    const float* k = ct + kKH * kDS + kh * kDS;
    float qr[4], kr[4];
    float qs = 0.0f, ks = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        qr[i] = q[lane + 32 * i];
        kr[i] = k[lane + 32 * i];
        qs += qr[i] * qr[i];
        ks += kr[i] * kr[i];
    }
    qs = warp_sum(qs);
    ks = warp_sum(ks);
    const float qn = rsqrtf(qs + 1e-6f) * rsqrtf((float)kDS);
    const float kn = rsqrtf(ks + 1e-6f);
    const float beta = sigmoidf_(ba[t * 64 + h]);
    const float a = ba[t * 64 + kVH + h] + dt_bias[h];
    const float sp = a > 20.0f ? a : log1pf(__expf(a));
    const float decay = __expf(sp * ssm_a[h]);
    if (lane == 0) factors[t * kVH + h] = make_float4(qn, kn, beta, decay);
}

template <int NT>
__global__ void __launch_bounds__(128) gdn_recur_prepared_kernel(const float* __restrict__ conv, const float4* __restrict__ factors,
                                                        float* __restrict__ state, float* __restrict__ o,
                                                        float* __restrict__ snap, int64_t snap_stride) {
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = blockIdx.y * 4 + warp;
    const int kh = h % kKH;
    float* S = state + ((size_t)h * kDS + col) * kDS;
    float s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = S[lane + 32 * i];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float* ct = conv + t * kConv;
        const float* q = ct + kh * kDS;
        const float* k = ct + kKH * kDS + kh * kDS;
        const float v = ct[2 * kKH * kDS + h * kDS + col];
        float qr[4], kr[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            qr[i] = q[lane + 32 * i];
            kr[i] = k[lane + 32 * i];
        }
        const float4 f = factors[t * kVH + h];
        const float qn = f.x, kn = f.y, beta = f.z, decay = f.w;
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            kr[i] *= kn;
            kv += s[i] * kr[i];
        }
        kv = warp_sum(kv);
        const float delta = (v - decay * kv) * beta;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] = decay * s[i] + kr[i] * delta;
            out += s[i] * qr[i];
        }
        out = warp_sum(out) * qn;
        if (lane == 0) o[t * 4096 + h * kDS + col] = out;
        if (snap && t + 1 < NT) {
            float* sn = snap + t * snap_stride + ((size_t)h * kDS + col) * kDS;
#pragma unroll
            for (int i = 0; i < 4; ++i) sn[lane + 32 * i] = s[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) S[lane + 32 * i] = s[i];
}

/// y = rmsnorm(o_h) * w * silu(z_h), quantized.  grid (32 heads, T) x 128 threads.
__global__ void gdn_norm_gate_kernel(const float* __restrict__ o, const float* __restrict__ qkvz, const float* __restrict__ nw,
                                     float eps, ActQ yq) {
    __shared__ float sh[32];
    const int h = blockIdx.x, tk = blockIdx.y, t = threadIdx.x;
    const float v = o[tk * 4096 + h * kDS + t];
    const float ss = block_sum(v * v, sh);
    const float r = rsqrtf(ss / kDS + eps);
    const float z = qkvz[tk * kQKVZ + kConv + h * kDS + t];
    quant_lane(v * r * nw[t] * silu(z), yq, tk * 4096 + h * kDS + t);
}

// =====================================================================================================================
// gated attention
// =====================================================================================================================
__device__ __forceinline__ void rope_neox64(float* sv /* smem, 256 */, int t, int pos, float base) {
    // pairs (i, i+32) for i < 32; theta = pos * base^(-2i/64).  Computed in double: fast-math sincosf is
    // inaccurate for the large angles long contexts produce.
    if (t < 32) {
        const double inv = pow((double)base, -2.0 * t / 64.0);
        double sn, cs;
        sincos((double)pos * (double)(float)inv, &sn, &cs);
        const float x0 = sv[t], x1 = sv[t + 32];
        sv[t] = x0 * (float)cs - x1 * (float)sn;
        sv[t + 32] = x0 * (float)sn + x1 * (float)cs;
    }
}

/// grid (20, T): blocks 0..15: q heads, 16..17: k heads, 18..19: v heads.  256 threads.
__global__ void __launch_bounds__(256) attn_prep_kernel(const float* __restrict__ qkv, const float* __restrict__ qnw,
                                                        const float* __restrict__ knw, float eps, const StepParams* P,
                                                        float rope_base, float* __restrict__ q_out,
                                                        __half* __restrict__ kc, __half* __restrict__ vc, int max_ctx) {
    __shared__ float sh[32];
    __shared__ float sv[kHD];
    const int b = blockIdx.x, t = threadIdx.x, tk = blockIdx.y;
    const int pos = P->pos + tk;
    qkv += tk * kQKV;
    q_out += tk * kNH * kHD;
    if (b >= kNH + kNKV) {
        const int kh = b - kNH - kNKV;
        vc[((size_t)kh * max_ctx + pos) * kHD + t] = __float2half(qkv[kNH * 2 * kHD + kNKV * kHD + kh * kHD + t]);
        return;
    }
    const bool isq = b < kNH;
    const float x = isq ? qkv[b * 2 * kHD + t] : qkv[kNH * 2 * kHD + (b - kNH) * kHD + t];
    const float ss = block_sum(x * x, sh);
    const float r = rsqrtf(ss / kHD + eps);
    sv[t] = x * r * (isq ? qnw[t] : knw[t]);
    __syncthreads();
    rope_neox64(sv, t, pos, rope_base);
    __syncthreads();
    if (isq) q_out[b * kHD + t] = sv[t] * 0.0625f;   // 1/sqrt(256)
    else kc[((size_t)(b - kNH) * max_ctx + pos) * kHD + t] = __float2half(sv[t]);
}

/// Keys per split for a context of L keys: a multiple of kAttSub, at most kAttSplits splits.
__device__ __forceinline__ int attn_chunk(int L) {
    const int per = (L + kAttSplits - 1) / kAttSplits;
    return (per + kAttSub - 1) / kAttSub * kAttSub;
}

/// grid (kAttSplits, 2 kv heads, T), 256 threads.  Flash-decoding: the block walks its key range in steps of 64 keys
/// with an online softmax for the 8 q heads that share this kv head.  Scoring: 4 threads per key (64 dims each);
/// accumulation: thread t owns output dim t.
__global__ void __launch_bounds__(256) attn_decode_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                          const __half* __restrict__ vc, const StepParams* P, int max_ctx,
                                                          float* __restrict__ part_o, float* __restrict__ part_ml) {
    __shared__ __align__(16) float sq[8][kHD];
    __shared__ float sp[8][kAttSub];
    __shared__ float sm[8], sl[8], sscale[8];
    const int split = blockIdx.x, kh = blockIdx.y, t = threadIdx.x, tk = blockIdx.z;
    const int L = P->pos + tk + 1;
    const int chunk = attn_chunk(L);
    const int j0 = split * chunk;
    if (j0 >= L) return;
    q += tk * kNH * kHD;
    part_o += (size_t)tk * kNH * kAttSplits * kHD;
    part_ml += (size_t)tk * kNH * kAttSplits * 2;
    const int j1 = min(j0 + chunk, L);
    for (int i = t; i < 8 * kHD; i += 256) sq[i / kHD][i % kHD] = q[(kh * 8) * kHD + i];
    if (t < 8) { sm[t] = -INFINITY; sl[t] = 0.0f; }
    __syncthreads();
    const int key = t >> 2, part = t & 3;   // scoring layout
    const int w = t >> 5, lane = t & 31;    // softmax layout: warp w owns head w
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s0 = j0; s0 < j1; s0 += kAttSub) {
        const int n = min(kAttSub, j1 - s0);
        float sc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (key < n) {
            const int4* kr = (const int4*)(kc + ((size_t)kh * max_ctx + s0 + key) * kHD + part * 64);
#pragma unroll 2
            for (int c = 0; c < 8; ++c) {
                const int4 raw = kr[c];
                const __half2* hp = (const __half2*)&raw;
                float kf[8];
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float2 f = __half22float2(hp[u]);
                    kf[2 * u] = f.x;
                    kf[2 * u + 1] = f.y;
                }
                const int d = part * 64 + c * 8;
#pragma unroll
                for (int h = 0; h < 8; ++h) {
                    const float4 q0 = *(const float4*)&sq[h][d], q1 = *(const float4*)&sq[h][d + 4];
                    sc[h] += kf[0] * q0.x + kf[1] * q0.y + kf[2] * q0.z + kf[3] * q0.w + kf[4] * q1.x + kf[5] * q1.y +
                             kf[6] * q1.z + kf[7] * q1.w;
                }
            }
        }
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            sc[h] += __shfl_xor_sync(0xffffffffu, sc[h], 1);
            sc[h] += __shfl_xor_sync(0xffffffffu, sc[h], 2);
        }
        if (part == 0) {
#pragma unroll
            for (int h = 0; h < 8; ++h) sp[h][key] = key < n ? sc[h] : -INFINITY;
        }
        __syncthreads();
        {   // online softmax update for head w (2 keys per lane)
            const float a0 = sp[w][lane], a1 = sp[w][lane + 32];
            const float mo = sm[w];
            const float mn = fmaxf(mo, warp_max(fmaxf(a0, a1)));   // finite: every step has at least one key
            const float e0 = __expf(a0 - mn), e1 = __expf(a1 - mn);
            sp[w][lane] = e0;
            sp[w][lane + 32] = e1;
            const float sum = warp_sum(e0 + e1);
            if (lane == 0) {
                const float f = __expf(mo - mn);   // 0 on the first step (mo = -inf)
                sscale[w] = f;
                sl[w] = sl[w] * f + sum;
                sm[w] = mn;
            }
        }
        __syncthreads();
        const __half* vb = vc + ((size_t)kh * max_ctx + s0) * kHD + t;
#pragma unroll
        for (int h = 0; h < 8; ++h) acc[h] *= sscale[h];
#pragma unroll 8
        for (int j = 0; j < n; ++j) {
            const float v = __half2float(vb[(size_t)j * kHD]);
#pragma unroll
            for (int h = 0; h < 8; ++h) acc[h] += sp[h][j] * v;
        }
        __syncthreads();   // sp / sscale are rewritten by the next step
    }
#pragma unroll
    for (int h = 0; h < 8; ++h) {
        const int qh = kh * 8 + h;
        part_o[((size_t)qh * kAttSplits + split) * kHD + t] = acc[h];
        if (t == 0) {
            part_ml[((size_t)qh * kAttSplits + split) * 2 + 0] = sm[h];
            part_ml[((size_t)qh * kAttSplits + split) * 2 + 1] = sl[h];
        }
    }
}

/// grid (16 q heads, T) x 256 threads: merge the splits, apply the sigmoid output gate, quantize for o_proj.
__global__ void __launch_bounds__(256) attn_combine_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml,
                                                           const float* __restrict__ qkv, const StepParams* P, ActQ oq) {
    constexpr int max_splits = kAttSplits;
    const int h = blockIdx.x, t = threadIdx.x, tk = blockIdx.y;
    const int L = P->pos + tk + 1;
    const int chunk = attn_chunk(L);
    const int ns = (L + chunk - 1) / chunk;
    part_o += (size_t)tk * kNH * max_splits * kHD;
    part_ml += (size_t)tk * kNH * max_splits * 2;
    qkv += tk * kQKV;
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, part_ml[((size_t)h * max_splits + s) * 2]);
    float o = 0.0f, l = 0.0f;
    for (int s = 0; s < ns; ++s) {
        const float f = __expf(part_ml[((size_t)h * max_splits + s) * 2] - M);
        l += f * part_ml[((size_t)h * max_splits + s) * 2 + 1];
        o += f * part_o[((size_t)h * max_splits + s) * kHD + t];
    }
    const float g = qkv[h * 2 * kHD + kHD + t];
    quant_lane(o / l * sigmoidf_(g), oq, tk * kNH * kHD + h * kHD + t);
}

// =====================================================================================================================
// head / sampling
// =====================================================================================================================
constexpr int kArgmaxBlocks = 128;
__device__ float g_argmax_v[kMaxT][kArgmaxBlocks];
__device__ int g_argmax_i[kMaxT][kArgmaxBlocks];

__device__ __forceinline__ void argmax_merge(float& v, int& idx, float ov, int oi) {
    if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
}

__device__ float g_argmax_s[kMaxT][kArgmaxBlocks];

/// (max, argmax, sum of exp(x - max)) merge: the softmax denominator comes along with the argmax.
__device__ __forceinline__ void argmax_sum_merge(float& v, int& idx, float& s, float ov, int oi, float os) {
    if (ov == -INFINITY) return;
    if (v == -INFINITY) { v = ov; idx = oi; s = os; return; }
    const float m = fmaxf(v, ov);
    s = s * __expf(v - m) + os * __expf(ov - m);
    argmax_merge(v, idx, ov, oi);
}

/// Block reduction of (v, idx, optional s) over 256 threads; the result is valid in thread 0.
template <bool WithProb>
__device__ __forceinline__ void block_argmax_sum256(float& v, int& idx, float& s) {
    __shared__ float bv[8], bs[8];
    __shared__ int bi[8];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if constexpr (WithProb) {
            const float os = __shfl_xor_sync(0xffffffffu, s, o);
            argmax_sum_merge(v, idx, s, ov, oi, os);
        } else {
            argmax_merge(v, idx, ov, oi);
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        bv[warp] = v; bi[warp] = idx;
        if constexpr (WithProb) bs[warp] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0)
        for (int i = 1; i < 8; ++i) {
            if constexpr (WithProb) argmax_sum_merge(v, idx, s, bv[i], bi[i], bs[i]);
            else argmax_merge(v, idx, bv[i], bi[i]);
        }
}

/// grid (kArgmaxBlocks, T)
template <bool WithProb>
__global__ void __launch_bounds__(256) argmax_stage1(const float* __restrict__ x, int n) {
    const int row = blockIdx.y;
    x += (size_t)row * n;
    const int per = (n + kArgmaxBlocks - 1) / kArgmaxBlocks;
    const int i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    float v = -INFINITY, s = 0.0f;
    int idx = 0x7fffffff;
    for (int i = i0 + threadIdx.x; i < i1; i += 256) {   // online: s = sum exp(x - v) over what was seen
        const float xi = x[i];
        if (xi > v) {
            if constexpr (WithProb) s = (v == -INFINITY ? 0.0f : s * __expf(v - xi)) + 1.0f;
            v = xi;
            idx = i;
        } else {
            if constexpr (WithProb) s += __expf(xi - v);
        }
    }
    block_argmax_sum256<WithProb>(v, idx, s);
    if (threadIdx.x == 0) {
        g_argmax_v[row][blockIdx.x] = v; g_argmax_i[row][blockIdx.x] = idx;
        if constexpr (WithProb) g_argmax_s[row][blockIdx.x] = s;
    }
}

/// grid T.  prob (optional): softmax probability of the argmax.
template <bool WithProb>
__global__ void __launch_bounds__(256) argmax_stage2(int32_t* out, float* prob) {
    const int row = blockIdx.x;
    float v = -INFINITY, s = 0.0f;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < kArgmaxBlocks; i += 256) {
        if constexpr (WithProb)
            argmax_sum_merge(v, idx, s, g_argmax_v[row][i], g_argmax_i[row][i], g_argmax_s[row][i]);
        else
            argmax_merge(v, idx, g_argmax_v[row][i], g_argmax_i[row][i]);
    }
    block_argmax_sum256<WithProb>(v, idx, s);
    if (threadIdx.x == 0) {
        out[row] = idx;
        if constexpr (WithProb) prob[row] = 1.0f / s;
    }
}

constexpr int kCandBlocks = 128;
constexpr int kCandIPT = 2;   // items per thread, stage 1

__device__ __forceinline__ float penalize(float l, uint32_t c, SamplePen pen) {
    if (c == 0) return l;
    if (pen.repetition != 1.0f) l = l > 0.0f ? l / pen.repetition : l * pen.repetition;
    return l - pen.presence - pen.frequency * (float)c;
}

__global__ void __launch_bounds__(1024) cand_stage1(const float* __restrict__ logits, int n, const uint32_t* __restrict__ counts,
                                                    SamplePen pen, float* __restrict__ tmp) {
    using Sort = cub::BlockRadixSort<float, 1024, kCandIPT, int>;
    __shared__ typename Sort::TempStorage ts;
    float k[kCandIPT];
    int v[kCandIPT];
    const int per_block = (n + kCandBlocks - 1) / kCandBlocks;
    const int b0 = blockIdx.x * per_block;
#pragma unroll
    for (int i = 0; i < kCandIPT; ++i) {
        const int idx = b0 + threadIdx.x + i * 1024;
        if (idx < n && idx < b0 + per_block) {
            k[i] = counts ? penalize(logits[idx], counts[idx], pen) : logits[idx];
            v[i] = idx;
        } else {
            k[i] = -INFINITY;
            v[i] = -1;
        }
    }
    Sort(ts).SortDescendingBlockedToStriped(k, v);
    // striped: thread t holds ranks t, t+1024 -> the top kCand are threads 0..kCand-1, item 0
    if (threadIdx.x < kCand) {
        tmp[(blockIdx.x * kCand + threadIdx.x) * 2] = k[0];
        ((int*)tmp)[(blockIdx.x * kCand + threadIdx.x) * 2 + 1] = v[0];
    }
}

__global__ void __launch_bounds__(1024) cand_stage2(const float* __restrict__ tmp, Candidates* out) {
    constexpr int IPT = kCandBlocks * kCand / 1024;
    using Sort = cub::BlockRadixSort<float, 1024, IPT, int>;
    __shared__ typename Sort::TempStorage ts;
    float k[IPT];
    int v[IPT];
#pragma unroll
    for (int i = 0; i < IPT; ++i) {
        const int idx = threadIdx.x * IPT + i;
        k[i] = tmp[idx * 2];
        v[i] = ((const int*)tmp)[idx * 2 + 1];
    }
    Sort(ts).SortDescendingBlockedToStriped(k, v);
    if (threadIdx.x < kCand) {
        out->id[threadIdx.x] = v[0];
        out->logit[threadIdx.x] = k[0];
    }
}

// token[0]: the token entering the penalty window; token[1]: the one leaving it (-1: none)
__global__ void count_token_kernel(uint32_t* counts, const int32_t* token) {
    counts[token[0]] += 1;
    if (token[1] >= 0 && counts[token[1]] > 0) counts[token[1]] -= 1;
}

bool gpu_expert_reuse_enabled() {
    // The T=1 and legacy K-quant paths are unchanged. Pascal register pressure must be measured before
    // making reuse a default; a process-level opt-in also keeps captured launch selection stable.
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GPU_EXPERT_REUSE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

bool gpu_expert_transpose_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GPU_EXPERT_TRANSPOSE");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

}  // namespace

// =====================================================================================================================
// launchers
// =====================================================================================================================
#define SQ_NT_SWITCH(T, CALL)                                                   \
    switch (T) {                                                                \
        case 1: { constexpr int NT = 1; CALL; } break;                          \
        case 2: { constexpr int NT = 2; CALL; } break;                          \
        case 3: { constexpr int NT = 3; CALL; } break;                          \
        case 4: { constexpr int NT = 4; CALL; } break;                          \
        default: die("step of %d tokens (max %d)", T, kMaxT);                   \
    }

#define SQ_MULTI_NT_SWITCH(T, CALL)                                             \
    switch (T) {                                                                \
        case 2: { constexpr int NT = 2; CALL; } break;                          \
        case 3: { constexpr int NT = 3; CALL; } break;                          \
        case 4: { constexpr int NT = 4; CALL; } break;                          \
        default: die("expert reuse requires 2..4 tokens, got %d", T);          \
    }

void k_embed_in(const StepParams* P, float* x, int T, cudaStream_t s) {
    embed_in_kernel<<<dim3(kE / 256, T), 256, 0, s>>>(P, x);
}

void k_rmsnorm_q(const float* x, const float* w, float eps, float* out_f, ActQ out_q, int T, cudaStream_t s) {
    rmsnorm_q_kernel<<<T, 256, 0, s>>>(x, w, eps, out_f, out_q);
}

void k_add_rmsnorm_q(float* x, const float* a, const float* w, float eps, float* xn, ActQ xq, int T, cudaStream_t s) {
    add_rmsnorm_q_kernel<<<T, 256, 0, s>>>(x, a, w, eps, xn, xq);
}

void k_mtp_in(const StepParams* P, const float* h, const float* enorm, const float* hnorm, float eps, ActQ out, int T,
              cudaStream_t s) {
    mtp_in_kernel<<<dim3(T, 2), 256, 0, s>>>(P, h, enorm, hnorm, eps, out);
}

void k_test_gemv_q8(const DQ8& W, ActQ a, float* y, int T, bool readonly, cudaStream_t s) {
    if (W.q6) { k_gemv_q6(W, a, y, T, s); return; }
    const int blocks = (W.rows + kGemvWarps - 1) / kGemvWarps;
    const size_t smem = ((size_t)W.cols + (size_t)(W.cols / 32) * 4) * T;
    if (readonly) {
        SQ_NT_SWITCH(T, (gemv_q8_kernel<NT, true><<<blocks, kGemvWarps * 32, smem, s>>>(W.qs, (const __half*)W.d, a, y, W.rows, W.cols)));
    } else {
        SQ_NT_SWITCH(T, (gemv_q8_kernel<NT, false><<<blocks, kGemvWarps * 32, smem, s>>>(W.qs, (const __half*)W.d, a, y, W.rows, W.cols)));
    }
}

void k_gemv_q8(const DQ8& W, ActQ a, float* y, int T, cudaStream_t s) {
    k_test_gemv_q8(W, a, y, T, actq_readonly_enabled(), s);
}

void k_gemv_f32(const DF32& W, const float* x, float* y, int T, cudaStream_t s) {
    SQ_CHECK(W.cols % 16 == 0, "gemv_f32: cols %d", W.cols);
    SQ_NT_SWITCH(T, (gemv_f32_kernel<NT><<<(W.rows + 1) / 2, 256, 0, s>>>(W.w, x, y, W.rows, W.cols)));
}

void k_gdn_conv(const float* qkvz, float* conv_state, const float* conv_w, float* conv_out, int T, float* snap,
                int64_t snap_stride, cudaStream_t s) {
    gdn_conv_kernel<<<kConv / 256, 256, 0, s>>>(qkvz, conv_state, conv_w, conv_out, T, snap, snap_stride);
}

bool gdn_prepare_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_GDN_PREPARE");
        if (!value) return false;
        SQ_CHECK(std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0,
                 "STRATA_GDN_PREPARE must be 0 or 1");
        return value[0] == '1';
    }();
    return enabled;
}

void k_gdn_recur(const float* conv_out, const float* ba, const float* dt_bias, const float* ssm_a, float* state,
                 float* o, int T, float* snap, int64_t snap_stride, cudaStream_t s, float* prepared_factors) {
    if (prepared_factors) {
        gdn_prepare_kernel<<<dim3(kVH, T), 32, 0, s>>>(conv_out, ba, dt_bias, ssm_a, (float4*)prepared_factors);
        SQ_NT_SWITCH(T, (gdn_recur_prepared_kernel<NT><<<dim3(kVH, kDS / 4), 128, 0, s>>>(
            conv_out, (const float4*)prepared_factors, state, o, snap, snap_stride)));
    } else {
        SQ_NT_SWITCH(T, (gdn_recur_kernel<NT><<<dim3(kVH, kDS / 4), 128, 0, s>>>(conv_out, ba, dt_bias, ssm_a, state, o, snap,
                                                                                 snap_stride)));
    }
}

void k_gdn_norm_gate(const float* o, const float* qkvz, const float* norm_w, float eps, ActQ yq, int T, cudaStream_t s) {
    gdn_norm_gate_kernel<<<dim3(kVH, T), kDS, 0, s>>>(o, qkvz, norm_w, eps, yq);
}

void k_attn_prep(const float* qkv, const float* q_norm, const float* k_norm, float eps, const StepParams* P,
                 float rope_base, float* q_out, uint16_t* k_cache, uint16_t* v_cache, int max_ctx, int T, cudaStream_t s) {
    attn_prep_kernel<<<dim3(kNH + 2 * kNKV, T), 256, 0, s>>>(qkv, q_norm, k_norm, eps, P, rope_base, q_out, (__half*)k_cache,
                                                             (__half*)v_cache, max_ctx);
}

int attn_max_splits(int) { return kAttSplits; }

void k_attn_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, const StepParams* P, int max_ctx,
                   float* part_o, float* part_ml, int T, cudaStream_t s) {
    attn_decode_kernel<<<dim3(kAttSplits, kNKV, T), 256, 0, s>>>(q, (const __half*)k_cache, (const __half*)v_cache, P,
                                                                 max_ctx, part_o, part_ml);
}

void k_attn_combine(const float* part_o, const float* part_ml, const float* qkv, const StepParams* P, int max_ctx,
                    ActQ oq, int T, cudaStream_t s) {
    (void)max_ctx;
    attn_combine_kernel<<<dim3(kNH, T), 256, 0, s>>>(part_o, part_ml, qkv, P, oq);
}

void k_test_router(const float* logits, int layer, const int32_t* residency, const CacheLayerInfo* cinfo, const float* xn,
              HitList* hits, Mailbox* mb_dev, const StepParams* P, int32_t* d_nmiss, float* d_sg, uint32_t* counts,
              unsigned long long* ecount, int T, cudaStream_t s, bool omit_system_fence) {
    // The fence-free specialization is valid only for ordinary device memory. The event path copies the
    // mailbox on this same stream after kernel completion, then waits for the copy's completion event.
    // Keep every block barrier and every store; only legacy CPU polling needs system visibility mid-kernel.
    if (omit_system_fence) {
        if (T == 1)
            router_kernel<false><<<1, 256, 0, s>>>(logits, layer, residency, cinfo, xn, hits, mb_dev, P, d_nmiss, d_sg, counts, ecount);
        else
            router_multi_kernel<false><<<1, 256, 0, s>>>(logits, layer, residency, cinfo, xn, hits, mb_dev, P, d_nmiss, d_sg, counts, ecount, T);
    } else {
        if (T == 1)
            router_kernel<true><<<1, 256, 0, s>>>(logits, layer, residency, cinfo, xn, hits, mb_dev, P, d_nmiss, d_sg, counts, ecount);
        else
            router_multi_kernel<true><<<1, 256, 0, s>>>(logits, layer, residency, cinfo, xn, hits, mb_dev, P, d_nmiss, d_sg, counts, ecount, T);
    }
}

void k_router(const float* logits, int layer, const int32_t* residency, const CacheLayerInfo* cinfo, const float* xn,
              HitList* hits, Mailbox* mb_dev, const StepParams* P, int32_t* d_nmiss, float* d_sg, uint32_t* counts,
              unsigned long long* ecount, int T, cudaStream_t s, bool device_mailbox) {
    static const bool device_fence_optimization = [] {
        const char* value = std::getenv("STRATA_ROUTER_DEVICE_FENCE");
        return value && std::atoi(value) != 0;
    }();
    k_test_router(logits, layer, residency, cinfo, xn, hits, mb_dev, P, d_nmiss, d_sg, counts, ecount, T, s,
                  device_mailbox && device_fence_optimization);
}

void k_swiglu_q(const float* gu, ActQ out, int n_ff, int T, cudaStream_t s) {
    swiglu_q_kernel<<<dim3((n_ff + 255) / 256, T), 256, 0, s>>>(gu, out, n_ff);
}

void k_test_moe_gu(uint32_t type, ActQ x, const HitList* hits, float* h_gu, int T, bool reuse, cudaStream_t s, bool transpose,
                   int known_experts) {
    SQ_CHECK(known_experts >= 0 && known_experts <= kMaxK * T, "invalid host-known expert launch bound");
    const dim3 grid(2 * kFF / 8, known_experts ? known_experts : kMaxK * T);
    if (transpose) {
#define SQ_TRANSPOSE_GU_CASE(Type) \
        case Type: \
            if (T > 1 && reuse) { SQ_MULTI_NT_SWITCH(T, (moe_gu_kernel<Type, NT, true, true><<<grid, 256, 0, s>>>(x, hits, h_gu))); } \
            else { SQ_NT_SWITCH(T, (moe_gu_kernel<Type, NT, false, true><<<grid, 256, 0, s>>>(x, hits, h_gu))); } \
            return
        switch (type) {
            SQ_TRANSPOSE_GU_CASE(T_Q2_K);
            SQ_TRANSPOSE_GU_CASE(T_Q3_K);
            SQ_TRANSPOSE_GU_CASE(T_IQ2_S);
            SQ_TRANSPOSE_GU_CASE(T_IQ3_S);
            SQ_TRANSPOSE_GU_CASE(T_IQ4_XS);
        }
#undef SQ_TRANSPOSE_GU_CASE
    }
    if (T > 1 && reuse) {
        switch (type) {
            case T_Q2_K: SQ_MULTI_NT_SWITCH(T, (moe_gu_kernel<T_Q2_K, NT, true><<<grid, 256, 0, s>>>(x, hits, h_gu))); return;
            case T_Q3_K: SQ_MULTI_NT_SWITCH(T, (moe_gu_kernel<T_Q3_K, NT, true><<<grid, 256, 0, s>>>(x, hits, h_gu))); return;
            case T_IQ2_S: SQ_MULTI_NT_SWITCH(T, (moe_gu_kernel<T_IQ2_S, NT, true><<<grid, 256, 0, s>>>(x, hits, h_gu))); return;
            case T_IQ3_S: SQ_MULTI_NT_SWITCH(T, (moe_gu_kernel<T_IQ3_S, NT, true><<<grid, 256, 0, s>>>(x, hits, h_gu))); return;
            case T_IQ4_XS: SQ_MULTI_NT_SWITCH(T, (moe_gu_kernel<T_IQ4_XS, NT, true><<<grid, 256, 0, s>>>(x, hits, h_gu))); return;
        }
    }
    switch (type) {
        case T_Q2_K: SQ_NT_SWITCH(T, (moe_gu_kernel<T_Q2_K, NT><<<grid, 256, 0, s>>>(x, hits, h_gu))); break;
        case T_Q3_K: SQ_NT_SWITCH(T, (moe_gu_kernel<T_Q3_K, NT><<<grid, 256, 0, s>>>(x, hits, h_gu))); break;
        case T_IQ2_S: SQ_NT_SWITCH(T, (moe_gu_kernel<T_IQ2_S, NT><<<grid, 256, 0, s>>>(x, hits, h_gu))); break;
        case T_IQ3_S: SQ_NT_SWITCH(T, (moe_gu_kernel<T_IQ3_S, NT><<<grid, 256, 0, s>>>(x, hits, h_gu))); break;
        case T_IQ4_XS: SQ_NT_SWITCH(T, (moe_gu_kernel<T_IQ4_XS, NT><<<grid, 256, 0, s>>>(x, hits, h_gu))); break;
        case T_Q4_K: SQ_NT_SWITCH(T, (moe_gu_kernel<T_Q4_K, NT><<<grid, 256, 0, s>>>(x, hits, h_gu))); break;
        case T_Q5_K: SQ_NT_SWITCH(T, (moe_gu_kernel<T_Q5_K, NT><<<grid, 256, 0, s>>>(x, hits, h_gu))); break;
        case T_Q6_K: SQ_NT_SWITCH(T, (moe_gu_kernel<T_Q6_K, NT><<<grid, 256, 0, s>>>(x, hits, h_gu))); break;
        default: die("moe_gu: unsupported type %u", type);
    }
}

void k_moe_gu(uint32_t type, ActQ x, const HitList* hits, float* h_gu, int T, cudaStream_t s, int known_experts) {
    k_test_moe_gu(type, x, hits, h_gu, T, gpu_expert_reuse_enabled(), s, gpu_expert_transpose_enabled(), known_experts);
}

void k_add_expert_output(float* destination, const float* source, int elements, cudaStream_t s) {
    add_expert_output_kernel<<<(elements + 255) / 256, 256, 0, s>>>(destination, source, elements);
}

void k_moe_act(const float* h_gu, const HitList* hits, ActQ hq, int T, cudaStream_t s) {
    moe_act_kernel<<<kMaxK * T, kFF, 0, s>>>(h_gu, hits, hq);
}

void k_test_moe_down(uint32_t type, int64_t down_off, ActQ hq, const HitList* hits, float* out, int T, bool reuse,
                     cudaStream_t s, bool transpose) {
    const int grid = kE / 16;
    if (transpose) {
#define SQ_TRANSPOSE_DOWN_CASE(Type) \
        case Type: \
            if (T > 1 && reuse) { SQ_MULTI_NT_SWITCH(T, (moe_down_kernel<Type, NT, true, true><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); } \
            else { SQ_NT_SWITCH(T, (moe_down_kernel<Type, NT, false, true><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); } \
            return
        switch (type) {
            SQ_TRANSPOSE_DOWN_CASE(T_Q2_K);
            SQ_TRANSPOSE_DOWN_CASE(T_Q3_K);
            SQ_TRANSPOSE_DOWN_CASE(T_IQ2_S);
            SQ_TRANSPOSE_DOWN_CASE(T_IQ3_S);
            SQ_TRANSPOSE_DOWN_CASE(T_IQ4_XS);
        }
#undef SQ_TRANSPOSE_DOWN_CASE
    }
    if (T > 1 && reuse) {
        switch (type) {
            case T_Q2_K: SQ_MULTI_NT_SWITCH(T, (moe_down_kernel<T_Q2_K, NT, true><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); return;
            case T_Q3_K: SQ_MULTI_NT_SWITCH(T, (moe_down_kernel<T_Q3_K, NT, true><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); return;
            case T_IQ2_S: SQ_MULTI_NT_SWITCH(T, (moe_down_kernel<T_IQ2_S, NT, true><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); return;
            case T_IQ3_S: SQ_MULTI_NT_SWITCH(T, (moe_down_kernel<T_IQ3_S, NT, true><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); return;
            case T_IQ4_XS: SQ_MULTI_NT_SWITCH(T, (moe_down_kernel<T_IQ4_XS, NT, true><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); return;
        }
    }
    switch (type) {
        case T_Q2_K: SQ_NT_SWITCH(T, (moe_down_kernel<T_Q2_K, NT><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); break;
        case T_Q3_K: SQ_NT_SWITCH(T, (moe_down_kernel<T_Q3_K, NT><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); break;
        case T_IQ2_S: SQ_NT_SWITCH(T, (moe_down_kernel<T_IQ2_S, NT><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); break;
        case T_IQ3_S: SQ_NT_SWITCH(T, (moe_down_kernel<T_IQ3_S, NT><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); break;
        case T_IQ4_XS: SQ_NT_SWITCH(T, (moe_down_kernel<T_IQ4_XS, NT><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); break;
        case T_Q4_K: SQ_NT_SWITCH(T, (moe_down_kernel<T_Q4_K, NT><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); break;
        case T_Q5_K: SQ_NT_SWITCH(T, (moe_down_kernel<T_Q5_K, NT><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); break;
        case T_Q6_K: SQ_NT_SWITCH(T, (moe_down_kernel<T_Q6_K, NT><<<grid, 256, 0, s>>>(down_off, hq, hits, out))); break;
        default: die("moe_down: unsupported type %u", type);
    }
}

void k_moe_down(uint32_t type, int64_t down_off, ActQ hq, const HitList* hits, float* out, int T, cudaStream_t s) {
    k_test_moe_down(type, down_off, hq, hits, out, T, gpu_expert_reuse_enabled(), s, gpu_expert_transpose_enabled());
}

void k_combine(float* x, const float* moe_gpu, const float* sh_out, const float* d_sg, const int32_t* d_nmiss,
               const Result* res_dev, const StepParams* P, const float* next_w, float eps, float* xn, ActQ xq,
               int32_t* err, unsigned long long* wait_ns, int T, cudaStream_t s) {
    combine_kernel<<<T, 256, 0, s>>>(x, moe_gpu, sh_out, d_sg, d_nmiss, res_dev, P, next_w, eps, xn, xq, err, wait_ns);
}

void k_argmax(const float* logits, int n, int32_t* out_dev, int T, cudaStream_t s, float* prob_dev) {
    // Trunk verification and ordinary greedy sampling need only an index. Keep the probability path's
    // strict comparisons, lower-id tie break, and sentinel for rows containing only NaN/-infinity.
    // MTP confidence still uses the original online softmax arithmetic when a probability is requested.
    if (prob_dev) {
        argmax_stage1<true><<<dim3(kArgmaxBlocks, T), 256, 0, s>>>(logits, n);
        argmax_stage2<true><<<T, 256, 0, s>>>(out_dev, prob_dev);
    } else {
        argmax_stage1<false><<<dim3(kArgmaxBlocks, T), 256, 0, s>>>(logits, n);
        argmax_stage2<false><<<T, 256, 0, s>>>(out_dev, nullptr);
    }
}

void k_candidates(const float* logits, int n, const uint32_t* counts, SamplePen pen, float* tmp, Candidates* out_dev,
                  cudaStream_t s) {
    SQ_CHECK((n + kCandBlocks - 1) / kCandBlocks <= 1024 * kCandIPT, "candidates: vocab too large");
    cand_stage1<<<kCandBlocks, 1024, 0, s>>>(logits, n, counts, pen, tmp);
    cand_stage2<<<1, 1024, 0, s>>>(tmp, out_dev);
}

void k_count_token(uint32_t* counts, const int32_t* token, cudaStream_t s) { count_token_kernel<<<1, 1, 0, s>>>(counts, token); }

void k_copy(void* dst, const void* src, size_t bytes, cudaStream_t s) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s));
}

}  // namespace sq
