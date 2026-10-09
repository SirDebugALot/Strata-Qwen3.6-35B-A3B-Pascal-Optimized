// Optional INT8 KV storage for Qwen's gated full-attention layers.
// The validated FP16 kernels in kernels.cu remain untouched. This branch keeps
// their RoPE, split attention, causal mask and FP32 softmax accumulation, changing
// only the stored K/V representation to signed bytes plus a FP32 scale per head
// and token. No tensor cores or instructions newer than Pascal are required.
#include "kernels.cuh"
#include "common.hpp"

#include <cuda_fp16.h>

namespace sq {
namespace {

constexpr int kHD = 256, kNH = 16, kNKV = 2;
constexpr int kQKV = kNH * 2 * kHD + 2 * kNKV * kHD;
constexpr int kAttSplits = 64, kAttSub = 64;

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
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = lane < nw ? sh[lane] : 0.0f;
    v = warp_sum(v);
    return v;
}
__device__ __forceinline__ float block_max(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    return warp_max(lane < nw ? sh[lane] : 0.0f);
}
__device__ __forceinline__ void rope_neox64(float* sv, int t, int pos, float base) {
    if (t < 32) {
        const double inv = pow((double)base, -2.0 * t / 64.0);
        double sn, cs;
        sincos((double)pos * (double)(float)inv, &sn, &cs);
        const float x0 = sv[t], x1 = sv[t + 32];
        sv[t] = x0 * (float)cs - x1 * (float)sn;
        sv[t + 32] = x0 * (float)sn + x1 * (float)cs;
    }
}

__device__ __forceinline__ void store_head(float value, int8_t* cache, float* scales,
                                          size_t head_token, int dim, float* scratch) {
    const float amax = block_max(fabsf(value), scratch);
    const float d = amax / 127.0f;
    const int q = amax > 0.0f ? __float2int_rn(value / d) : 0;
    cache[head_token * kHD + dim] = static_cast<int8_t>(max(-127, min(127, q)));
    if (dim == 0) scales[head_token] = d;
}

__global__ void __launch_bounds__(256) attn_prep_i8_kernel(
        const float* __restrict__ qkv, const float* __restrict__ qnw, const float* __restrict__ knw,
        float eps, const StepParams* P, float rope_base, float* __restrict__ q_out,
        int8_t* __restrict__ kc, int8_t* __restrict__ vc, float* __restrict__ kd, float* __restrict__ vd,
        int max_ctx) {
    __shared__ float sh[32];
    __shared__ float sv[kHD];
    const int b = blockIdx.x, t = threadIdx.x, tk = blockIdx.y;
    const int pos = P->pos + tk;
    qkv += tk * kQKV;
    q_out += tk * kNH * kHD;
    if (b >= kNH + kNKV) {
        const int kh = b - kNH - kNKV;
        const float value = qkv[kNH * 2 * kHD + kNKV * kHD + kh * kHD + t];
        store_head(value, vc, vd, (size_t)kh * max_ctx + pos, t, sh);
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
    if (isq) q_out[b * kHD + t] = sv[t] * 0.0625f;
    else store_head(sv[t], kc, kd, (size_t)(b - kNH) * max_ctx + pos, t, sh);
}

__device__ __forceinline__ int attn_chunk(int L) {
    const int per = (L + kAttSplits - 1) / kAttSplits;
    return (per + kAttSub - 1) / kAttSub * kAttSub;
}

// Same split boundaries and FP32 operation order as the FP16 attention decoder.
// A key's scale is reused for all 256 dimensions and its eight grouped Q heads.
__global__ void __launch_bounds__(256) attn_decode_i8_kernel(
        const float* __restrict__ q, const int8_t* __restrict__ kc, const int8_t* __restrict__ vc,
        const float* __restrict__ kd, const float* __restrict__ vd, const StepParams* P, int max_ctx,
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
    const int key = t >> 2, part = t & 3;
    const int w = t >> 5, lane = t & 31;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s0 = j0; s0 < j1; s0 += kAttSub) {
        const int n = min(kAttSub, j1 - s0);
        float sc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (key < n) {
            const size_t head_token = (size_t)kh * max_ctx + s0 + key;
            const int2* kr = reinterpret_cast<const int2*>(kc + head_token * kHD + part * 64);
            const float dkey = kd[head_token];
#pragma unroll 2
            for (int c = 0; c < 8; ++c) {
                const int2 raw = kr[c];
                const int8_t* qp = reinterpret_cast<const int8_t*>(&raw);
                float kf[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) kf[u] = static_cast<float>(qp[u]) * dkey;
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
        {
            const float a0 = sp[w][lane], a1 = sp[w][lane + 32];
            const float mo = sm[w];
            const float mn = fmaxf(mo, warp_max(fmaxf(a0, a1)));
            const float e0 = __expf(a0 - mn), e1 = __expf(a1 - mn);
            sp[w][lane] = e0;
            sp[w][lane + 32] = e1;
            const float sum = warp_sum(e0 + e1);
            if (lane == 0) {
                const float f = __expf(mo - mn);
                sscale[w] = f;
                sl[w] = sl[w] * f + sum;
                sm[w] = mn;
            }
        }
        __syncthreads();
        const size_t first = (size_t)kh * max_ctx + s0;
        const int8_t* vb = vc + first * kHD + t;
#pragma unroll
        for (int h = 0; h < 8; ++h) acc[h] *= sscale[h];
#pragma unroll 8
        for (int j = 0; j < n; ++j) {
            const float v = static_cast<float>(vb[(size_t)j * kHD]) * vd[first + j];
#pragma unroll
            for (int h = 0; h < 8; ++h) acc[h] += sp[h][j] * v;
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = 0; h < 8; ++h) {
        const int qh = kh * 8 + h;
        part_o[((size_t)qh * kAttSplits + split) * kHD + t] = acc[h];
        if (t == 0) {
            part_ml[((size_t)qh * kAttSplits + split) * 2] = sm[h];
            part_ml[((size_t)qh * kAttSplits + split) * 2 + 1] = sl[h];
        }
    }
}

}  // namespace

void k_attn_prep_i8(const float* qkv, const float* q_norm, const float* k_norm, float eps, const StepParams* P,
                    float rope_base, float* q_out, int8_t* k_cache, int8_t* v_cache,
                    float* k_scales, float* v_scales, int max_ctx, int T, cudaStream_t s) {
    SQ_CHECK(T >= 1 && T <= kMaxT, "INT8 KV prep: invalid token count %d", T);
    attn_prep_i8_kernel<<<dim3(kNH + 2 * kNKV, T), 256, 0, s>>>(
        qkv, q_norm, k_norm, eps, P, rope_base, q_out, k_cache, v_cache, k_scales, v_scales, max_ctx);
}

void k_attn_decode_i8(const float* q, const int8_t* k_cache, const int8_t* v_cache,
                      const float* k_scales, const float* v_scales, const StepParams* P, int max_ctx,
                      float* part_o, float* part_ml, int T, cudaStream_t s) {
    SQ_CHECK(T >= 1 && T <= kMaxT, "INT8 KV decode: invalid token count %d", T);
    attn_decode_i8_kernel<<<dim3(kAttSplits, kNKV, T), 256, 0, s>>>(
        q, k_cache, v_cache, k_scales, v_scales, P, max_ctx, part_o, part_ml);
}

}  // namespace sq
